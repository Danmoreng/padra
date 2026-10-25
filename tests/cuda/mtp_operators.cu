#include "cuda/inference.cu"
#include <random>
namespace mellum16 {
void RequireProbe(bool x, const char *m) {
  if (!x)
    throw std::runtime_error(m);
}
void CacheProbe() {
  Stream stream;
  Buffer key(16 * 1024), value(16 * 1024), scales(16 * 32),
      binding(mtp::kCaches * sizeof(mtp::CacheBinding)), state(sizeof(DecodeState)),
      backup(mtp::kCaches * mtp::kMaxRows * 2048), bad(4);
  int cases = 0;
  for (int rowbytes : {1024, 512, 544})
    for (int position : {0, 15, 16, 31, 8187})
      for (int rows = 1; rows <= 5; ++rows)
        for (int retained = 0; retained <= rows; ++retained) {
          const int sb = rowbytes == 512 ? 32 : 0;
          mtp::CacheBinding bind{key.p, value.p, 16, sb ? scales.p : nullptr, rowbytes, sb};
          DecodeState ds{0, position, 8192, 5, position, std::min(position, 1024), 0, 0};
          std::vector<unsigned char> k(16 * rowbytes), v(k.size()), s(16 * sb), oldk, oldv, olds;
          for (std::size_t i = 0; i < k.size(); ++i) {
            k[i] = (i * 17 + 33) % 256;
            v[i] = (i * 23 + 77) % 256;
          }
          for (std::size_t i = 0; i < s.size(); ++i)
            s[i] = (i * 3 + 11) % 256;
          oldk = k;
          oldv = v;
          olds = s;
          Check(cudaMemcpyAsync(key.p, k.data(), k.size(), cudaMemcpyHostToDevice, stream.s));
          Check(cudaMemcpyAsync(value.p, v.data(), v.size(), cudaMemcpyHostToDevice, stream.s));
          if (sb)
            Check(cudaMemcpyAsync(scales.p, s.data(), s.size(), cudaMemcpyHostToDevice, stream.s));
          Check(cudaMemcpyAsync(binding.p, &bind, sizeof(bind), cudaMemcpyHostToDevice, stream.s));
          Check(cudaMemcpyAsync(state.p, &ds, sizeof(ds), cudaMemcpyHostToDevice, stream.s));
          Check(cudaMemsetAsync(bad.p, 0, 4, stream.s));
          Check(cudaMemsetAsync(backup.p, 0xaa, backup.bytes, stream.s));
          mtp::CopyCacheRows<false><<<dim3(rows, 1), 128, 0, stream.s>>>(
              binding.As<mtp::CacheBinding>(), state.As<DecodeState>(), backup.As<__nv_bfloat16>(),
              rows, retained, bad.As<int>());
          Check(cudaStreamSynchronize(stream.s));
          for (int r = 0; r < rows; ++r) {
            const int slot = (position + r) % 16;
            std::fill_n(k.begin() + slot * rowbytes, rowbytes, 0xee);
            std::fill_n(v.begin() + slot * rowbytes, rowbytes, 0xdd);
            if (sb)
              std::fill_n(s.begin() + slot * sb, sb, 0xcc);
          }
          Check(cudaMemcpyAsync(key.p, k.data(), k.size(), cudaMemcpyHostToDevice, stream.s));
          Check(cudaMemcpyAsync(value.p, v.data(), v.size(), cudaMemcpyHostToDevice, stream.s));
          if (sb)
            Check(cudaMemcpyAsync(scales.p, s.data(), s.size(), cudaMemcpyHostToDevice, stream.s));
          mtp::CopyCacheRows<true><<<dim3(rows, 1), 128, 0, stream.s>>>(
              binding.As<mtp::CacheBinding>(), state.As<DecodeState>(), backup.As<__nv_bfloat16>(),
              rows, retained, bad.As<int>());
          std::vector<unsigned char> actualk(k.size()), actualv(v.size()), actuals(s.size());
          int fault = 0;
          Check(cudaMemcpyAsync(actualk.data(), key.p, k.size(), cudaMemcpyDeviceToHost, stream.s));
          Check(
              cudaMemcpyAsync(actualv.data(), value.p, v.size(), cudaMemcpyDeviceToHost, stream.s));
          if (sb)
            Check(cudaMemcpyAsync(actuals.data(), scales.p, s.size(), cudaMemcpyDeviceToHost,
                                  stream.s));
          Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
          Check(cudaStreamSynchronize(stream.s));
          RequireProbe(!fault, "valid cache descriptor rejected");
          for (int r = retained; r < rows; ++r) {
            const int slot = (position + r) % 16;
            for (int i = 0; i < rowbytes; ++i) {
              k[slot * rowbytes + i] = position + r >= 16 ? oldk[slot * rowbytes + i] : 0;
              v[slot * rowbytes + i] = position + r >= 16 ? oldv[slot * rowbytes + i] : 0;
            }
            for (int i = 0; i < sb; ++i)
              s[slot * sb + i] = position + r >= 16 ? olds[slot * sb + i] : 0;
          }
          RequireProbe(k == actualk && v == actualv && s == actuals,
                       "encoded KV rollback differs from byte oracle");
          ++cases;
        }
  for (int malformed = 0; malformed < 7; ++malformed) {
    mtp::CacheBinding bind{key.p, value.p, 16, nullptr, 1024, 0};
    DecodeState ds{0, 0, 8192, 5, 0, 0, 0, 0};
    int rows = 2, retained = 0, layers = 1;
    if (malformed == 0)
      bind.row_bytes = 4096;
    if (malformed == 1)
      bind.scales = scales.p;
    if (malformed == 2)
      bind.capacity = 0;
    if (malformed == 3)
      rows = 6;
    if (malformed == 4)
      retained = 3;
    if (malformed == 5)
      ds.position = -1;
    if (malformed == 6)
      layers = 30;
    Check(cudaMemcpyAsync(binding.p, &bind, sizeof(bind), cudaMemcpyHostToDevice, stream.s));
    Check(cudaMemcpyAsync(state.p, &ds, sizeof(ds), cudaMemcpyHostToDevice, stream.s));
    Check(cudaMemsetAsync(bad.p, 0, 4, stream.s));
    // Every valid layer receives a descriptor; the final block must reject layer29 before reading
    // it.
    if (layers == 30) {
      std::array<mtp::CacheBinding, mtp::kCaches> all;
      all.fill(bind);
      Check(cudaMemcpyAsync(binding.p, all.data(), sizeof(all), cudaMemcpyHostToDevice, stream.s));
      mtp::CopyCacheRows<false><<<dim3(1, 30), 128, 0, stream.s>>>(
          binding.As<mtp::CacheBinding>(), state.As<DecodeState>(), backup.As<__nv_bfloat16>(), 2,
          0, bad.As<int>());
    } else
      mtp::CopyCacheRows<false><<<dim3(1, 1), 128, 0, stream.s>>>(
          binding.As<mtp::CacheBinding>(), state.As<DecodeState>(), backup.As<__nv_bfloat16>(),
          rows, retained, bad.As<int>());
    int fault = 0;
    Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
    Check(cudaStreamSynchronize(stream.s));
    RequireProbe(fault == 1, "bad cache descriptor accepted");
  }
  std::cout << "CACHE " << cases << " exact cases and 7 invalid guards passed\n";
}
void ExpertProbe(const std::filesystem::path &target, int rate) {
  auto loaded = internal::LoadCompiledModel(target);
  if (!loaded.ok())
    throw std::runtime_error(loaded.status().message());
  auto model = std::move(loaded).value();
  Stream stream;
  Buffer arena(model.weights.size());
  Check(cudaMemcpyAsync(arena.p, model.weights.data(), arena.bytes, cudaMemcpyHostToDevice,
                        stream.s));
  InitializeCodecTables();
  std::vector<DeviceMatrix> matrices;
  for (int e = 0; e < 64; ++e)
    for (const char *part : {"gate", "up", "down"})
      matrices.push_back(BindDeviceMatrix(model, arena.As<unsigned char>(),
                                          "model.layers.0.mlp.experts." + std::to_string(e) + "." +
                                              part + "_proj.weight"));
  Buffer descriptors(matrices.size() * sizeof(DeviceMatrix));
  Check(cudaMemcpyAsync(descriptors.p, matrices.data(), descriptors.bytes, cudaMemcpyHostToDevice,
                        stream.s));
  Buffer ids(40 * 4), masks(64 * 4), slots(64 * 5 * 4), active(64 * 4), count(4), bad(4),
      input(5 * 8 * 2304 * 4), a(5 * 8 * 2304 * 4), au(5 * 8 * 896 * 4), b(a.bytes), bu(au.bytes);
  std::vector<float> x(input.bytes / 4);
  std::mt19937 random(27);
  for (auto &f : x) {
    auto bits = std::uint16_t(0x3d00 + (random() % 1024));
    f = std::bit_cast<float>(std::uint32_t(bits) << 16) * (random() % 2 ? -1 : 1);
  }
  Check(cudaMemcpyAsync(input.p, x.data(), input.bytes, cudaMemcpyHostToDevice, stream.s));
  Check(cudaStreamSynchronize(stream.s));
  int tested = 0;
  for (int rows = 2; rows <= 5; ++rows)
    for (int route = 0; route < 8; ++route)
      for (int part : {0, 2}) {
        std::array<int, 40> selected{};
        for (int r = 0; r < rows; ++r)
          for (int s = 0; s < 8; ++s)
            selected[r * 8 + s] = (route * 8 + s + (route % 3) * r * 8) % 64;
        Check(cudaMemcpyAsync(ids.p, selected.data(), ids.bytes, cudaMemcpyHostToDevice, stream.s));
        Check(cudaMemsetAsync(bad.p, 0, 4, stream.s));
        mtp_experts::BuildUnion<true><<<1, 64, 0, stream.s>>>(ids.As<int>(), masks.As<unsigned>(),
                                                              slots.As<int>(), rows, bad.As<int>(),
                                                              active.As<int>(), count.As<int>());
        mtp_experts::Launch(descriptors.As<DeviceMatrix>(), 0, part, masks.As<unsigned>(),
                            slots.As<int>(), rows, input.As<float>(), b.As<float>(), bu.As<float>(),
                            bad.As<int>(), stream.s, active.As<int>(), count.As<int>(), rate);
        const int width = part == 2 ? 2304 : 896, istride = part == 2 ? 8 * 896 : 2304;
        for (int r = 0; r < rows; ++r) {
          const dim3 grid(width / 128, 8, part == 0 ? 2 : 1);
          if (rate == 3)
            exl3_mma_dual::Projection<3><<<grid, 256, 0, stream.s>>>(
                descriptors.As<DeviceMatrix>(), 0, part, ids.As<int>() + r * 8,
                input.As<float>() + r * istride, a.As<float>() + r * 8 * width,
                au.As<float>() + r * 8 * 896, 8, bad.As<int>());
          else if (rate == 4)
            exl3_mma_dual::Projection<4><<<grid, 256, 0, stream.s>>>(
                descriptors.As<DeviceMatrix>(), 0, part, ids.As<int>() + r * 8,
                input.As<float>() + r * istride, a.As<float>() + r * 8 * width,
                au.As<float>() + r * 8 * 896, 8, bad.As<int>());
          else
            exl3_mma_dual::Projection<6><<<grid, 256, 0, stream.s>>>(
                descriptors.As<DeviceMatrix>(), 0, part, ids.As<int>() + r * 8,
                input.As<float>() + r * istride, a.As<float>() + r * 8 * width,
                au.As<float>() + r * 8 * 896, 8, bad.As<int>());
        }
        std::vector<float> expected(rows * 8 * width), actual(expected.size());
        int fault = 0;
        Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
        Check(cudaMemcpyAsync(expected.data(), a.p, expected.size() * 4, cudaMemcpyDeviceToHost,
                              stream.s));
        Check(cudaMemcpyAsync(actual.data(), b.p, actual.size() * 4, cudaMemcpyDeviceToHost,
                              stream.s));
        Check(cudaStreamSynchronize(stream.s));
        RequireProbe(!fault && !std::memcmp(actual.data(), expected.data(), actual.size() * 4),
                     "MTP/ordinary expert projection mismatch");
        if (part == 0) {
          Check(cudaMemcpyAsync(expected.data(), au.p, expected.size() * 4, cudaMemcpyDeviceToHost,
                                stream.s));
          Check(cudaMemcpyAsync(actual.data(), bu.p, actual.size() * 4, cudaMemcpyDeviceToHost,
                                stream.s));
          Check(cudaStreamSynchronize(stream.s));
          RequireProbe(!std::memcmp(actual.data(), expected.data(), actual.size() * 4),
                       "MTP/ordinary up projection mismatch");
        }
        ++tested;
      }
  std::cout << "EXPERT K" << rate << ' ' << tested << " cases exact\n";
}
} // namespace mellum16
#include "checkpoint_probe.cuh"
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--checkpoint") {
      mellum16::CheckpointProbe();
      return 0;
    }
    mellum16::CacheProbe();
    if (argc == 3)
      mellum16::ExpertProbe(argv[1], std::stoi(argv[2]));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
