#pragma once
namespace mellum16 {
void Q8MtpAttentionProbe(bool quick = false) {
  constexpr int capacity = 131072;
  Stream stream;
  Buffer query(2 * 4096 * 4), encodedQuery(2 * 128 * sizeof(q8_native::QueryBlock)),
      state(2 * sizeof(DecodeState)), bad(4), saved(2 * sizeof(q8_native::CacheRow)),
      scratch(2 * 32 * 32 * 129 * 4), merged(2 * 4096 * 2), reference(2 * 4096 * 4),
      packed(2 * 4096), referencePacked(2 * 4096), scales(8), referenceScales(8);
  std::vector<float> queries(2 * 4096), expected(2 * 4096);
  std::vector<__nv_bfloat16> actual(2 * 4096);
  std::vector<unsigned char> expectedCodes(2 * 4096), actualCodes(2 * 4096);
  const auto row = [](int seed) {
    q8_native::CacheRow result;
    for (int d = 0; d < 16; ++d)
      result.d[d] = __float2half_rn(float(1 + (seed + d * 7) % 31) / 512.f);
    for (int d = 0; d < 512; ++d)
      result.qs[d] = static_cast<signed char>((seed * 13 + d * 19) % 255 - 127);
    return result;
  };
  for (int i = 0; i < 8192; ++i)
    queries[i] = __bfloat162float(__float2bfloat16_rn(float((i * 23) % 251 - 125) / 43.f));
  Check(cudaMemcpyAsync(query.p, queries.data(), query.bytes, cudaMemcpyHostToDevice, stream.s));
  int cases = 0, replays = 0;
  for (bool local : {false, true}) {
    const int cap = local ? 1024 : capacity, splits = local ? 16 : 32;
    Buffer keys(cap * sizeof(q8_native::CacheRow)), values(keys.bytes);
    std::vector<q8_native::CacheRow> historyK(cap), historyV(cap);
    for (int i = 0; i < cap; ++i) {
      historyK[i] = row(i + 1);
      historyV[i] = row(i + 71);
    }
    for (int position :
         {0, 1, 126, 127, 128, 1022, 1023, 1024, 4094, 4095, 4096, 16382, 32766, 65534, 131070}) {
      // Racecheck instruments every shared-memory access. This subset still
      // crosses tile/ring boundaries and reuses both async staging buffers.
      if (quick && !(local ? position == 1023 || position == 1024 || position == 131070
                           : position == 127 || position == 1024 || position == 4096))
        continue;
      Check(cudaMemcpyAsync(keys.p, historyK.data(), keys.bytes, cudaMemcpyHostToDevice, stream.s));
      Check(cudaMemcpyAsync(values.p, historyV.data(), values.bytes, cudaMemcpyHostToDevice,
                            stream.s));
      q8_native::CacheRow old[2] = {historyK[(position + 1) % cap], historyV[(position + 1) % cap]};
      Check(cudaMemcpyAsync(saved.p, old, sizeof(old), cudaMemcpyHostToDevice, stream.s));
      DecodeState states[2];
      for (int r = 0; r < 2; ++r)
        states[r] = {
            0, position + r, capacity, 2 - r, position + r, std::min(position + r, 1024), 0, 0};
      Check(cudaMemcpyAsync(state.p, states, sizeof(states), cudaMemcpyHostToDevice, stream.s));
      Check(cudaMemsetAsync(bad.p, 0, 4, stream.s));
      for (int r = 0; r < 2; ++r) {
        const auto k = row(position + r + 200003), v = row(position + r + 300007);
        Check(cudaMemcpyAsync(keys.As<q8_native::CacheRow>() + (position + r) % cap, &k, sizeof(k),
                              cudaMemcpyHostToDevice, stream.s));
        Check(cudaMemcpyAsync(values.As<q8_native::CacheRow>() + (position + r) % cap, &v,
                              sizeof(v), cudaMemcpyHostToDevice, stream.s));
        q8_native::EncodeQuery<><<<8, 128, 0, stream.s>>>(
            query.As<float>() + r * 4096, encodedQuery.As<q8_native::QueryBlock>(), bad.As<int>());
        auto *lse = scratch.As<float>() + splits * 32 * 128;
        if (local) {
          q8_native::Split<true><<<dim3(8, splits), 128, 0, stream.s>>>(
              encodedQuery.As<q8_native::QueryBlock>(), keys.As<q8_native::CacheRow>(),
              values.As<q8_native::CacheRow>(), scratch.As<float>(), lse,
              state.As<DecodeState>() + r, cap, splits, bad.As<int>(), capacity);
          q8_native::Merge<true, float><<<8, 128, 0, stream.s>>>(
              scratch.As<float>(), lse, reference.As<float>() + r * 4096,
              state.As<DecodeState>() + r, splits, bad.As<int>(), capacity);
        } else {
          q8_native::Split<false><<<dim3(8, splits), 128, 0, stream.s>>>(
              encodedQuery.As<q8_native::QueryBlock>(), keys.As<q8_native::CacheRow>(),
              values.As<q8_native::CacheRow>(), scratch.As<float>(), lse,
              state.As<DecodeState>() + r, cap, splits, bad.As<int>(), capacity);
          q8_native::Merge<false, float><<<8, 128, 0, stream.s>>>(
              scratch.As<float>(), lse, reference.As<float>() + r * 4096,
              state.As<DecodeState>() + r, splits, bad.As<int>(), capacity);
        }
        QuantizeFP8<<<1, 128, 0, stream.s>>>(reference.As<float>() + r * 4096,
                                             referencePacked.As<unsigned char>() + r * 4096,
                                             referenceScales.As<float>() + r, 4096, bad.As<int>());
        Check(cudaStreamSynchronize(stream.s)); // host row lifetime; no model-sized dump
      }
      Check(cudaMemcpyAsync(expected.data(), reference.p, reference.bytes, cudaMemcpyDeviceToHost,
                            stream.s));
      Check(cudaMemcpyAsync(expectedCodes.data(), referencePacked.p, referencePacked.bytes,
                            cudaMemcpyDeviceToHost, stream.s));
      float expectedScales[2], actualScales[2];
      Check(
          cudaMemcpyAsync(expectedScales, referenceScales.p, 8, cudaMemcpyDeviceToHost, stream.s));
      const auto launch = [&] {
        mtp_attention::LaunchQ8Native(
            query.As<float>(), keys.As<q8_native::CacheRow>(), values.As<q8_native::CacheRow>(),
            encodedQuery.As<q8_native::QueryBlock>(), merged.As<__nv_bfloat16>(),
            packed.As<unsigned char>(), scales.As<float>(), state.As<DecodeState>(), cap, local,
            scratch.As<float>(), saved.As<q8_native::CacheRow>(), bad.As<int>(), stream.s,
            capacity);
      };
      const auto verify = [&] {
        int fault = 0;
        Check(cudaMemcpyAsync(actual.data(), merged.p, merged.bytes, cudaMemcpyDeviceToHost,
                              stream.s));
        Check(cudaMemcpyAsync(actualCodes.data(), packed.p, packed.bytes, cudaMemcpyDeviceToHost,
                              stream.s));
        Check(cudaMemcpyAsync(actualScales, scales.p, 8, cudaMemcpyDeviceToHost, stream.s));
        Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
        Check(cudaStreamSynchronize(stream.s));
        RequireProbe(!fault, "Q8 batched attention rejected valid state");
        for (int i = 0; i < 8192; ++i)
          RequireProbe(__bfloat162float(actual[i]) == expected[i],
                       "Q8 batched attention value mismatch");
        RequireProbe(actualCodes == expectedCodes && !std::memcmp(actualScales, expectedScales, 8),
                     "Q8 batched attention FP8 code/scale mismatch");
      };
      Check(cudaMemsetAsync(scratch.p, 0xff, scratch.bytes, stream.s));
      launch();
      verify();
      ++cases;
      DecodeGraph graph;
      graph.Capture(stream.s, launch);
      for (int replay = 0; replay < 2; ++replay) {
        Check(cudaMemsetAsync(scratch.p, 0xff, scratch.bytes, stream.s));
        graph.Launch(stream.s);
        verify();
        ++replays;
      }
    }
  }
  std::cout << "Q8 MTP ATTENTION " << cases << " exact cases, " << replays
            << " graph replays passed\n";
}
} // namespace mellum16
