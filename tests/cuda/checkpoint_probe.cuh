#pragma once
namespace mellum16 {
void CheckpointProbe() {
  Stream stream;
  Buffer old(1024 * 1088), current(2048 * 1088), saved(old.bytes), command(sizeof(PrefillState)),
      bad(4);
  int cases = 0, graphs = 0;
  for (bool native : {false, true}) {
    const auto format = native ? KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2
                               : KVCacheFormat::Q8TokenHead128F32ScaleV1;
    const auto layout = KVCacheLayout::Make(format, 1024);
    for (int base : {0, 1, 255, 256, 1023, 1024, 2048, 4095, 129024})
      for (int tokens : {1, 4, 256, 2048}) {
        const auto chunk = KVCacheLayout::Make(format, tokens);
        for (int extent : {1, std::max(1, tokens / 2), tokens}) {
          const int anchor = base + extent;
          PrefillState state{base, base, tokens, 131072, base, 1, 1, 1};
          std::vector<unsigned char> history(layout.bytes), input(chunk.bytes),
              expected(layout.bytes), actual(layout.bytes);
          for (std::size_t i = 0; i < history.size(); ++i)
            history[i] = static_cast<unsigned char>(i * 17 + base * 3 + 11);
          for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<unsigned char>(i * 23 + anchor * 7 + 31);
          const auto plane = [&](std::size_t dst, std::size_t src, int width) {
            // Independent chronological oracle: write every valid retained row
            // into its physical ring slot. No modulo inverse from the kernel.
            for (int position = std::max(0, anchor - 1024); position < anchor; ++position) {
              const int slot = position % 1024;
              const auto *source = position < base ? history.data() + dst + slot * width
                                                   : input.data() + src + (position - base) * width;
              std::copy_n(source, width, expected.data() + dst + slot * width);
            }
          };
          plane(0, 0, native ? 544 : 512);
          plane(layout.v_offset, chunk.v_offset, native ? 544 : 512);
          if (!native)
            plane(layout.scales_offset, chunk.scales_offset, 32);
          Check(cudaMemcpyAsync(old.p, history.data(), history.size(), cudaMemcpyHostToDevice,
                                stream.s));
          Check(cudaMemcpyAsync(current.p, input.data(), input.size(), cudaMemcpyHostToDevice,
                                stream.s));
          Check(
              cudaMemcpyAsync(command.p, &state, sizeof(state), cudaMemcpyHostToDevice, stream.s));
          Check(cudaMemsetAsync(bad.p, 0, 4, stream.s));
          const auto launch = [&] {
            const auto *oldV = old.As<unsigned char>() + layout.v_offset;
            const auto *curV = current.As<unsigned char>() + chunk.v_offset;
            if (native)
              kv_checkpoint::CaptureEncoded<true>
                  <<<kv_checkpoint::Blocks, kv_checkpoint::Threads, 0, stream.s>>>(
                      old.p, oldV, nullptr, current.p, curV, nullptr, saved.p,
                      command.As<PrefillState>(), tokens, anchor, bad.As<int>());
            else
              kv_checkpoint::CaptureEncoded<false>
                  <<<kv_checkpoint::Blocks, kv_checkpoint::Threads, 0, stream.s>>>(
                      old.p, oldV, old.As<unsigned char>() + layout.scales_offset, current.p, curV,
                      current.As<unsigned char>() + chunk.scales_offset, saved.p,
                      command.As<PrefillState>(), tokens, anchor, bad.As<int>());
          };
          const auto verify = [&] {
            int fault = 0;
            Check(cudaMemcpyAsync(actual.data(), saved.p, actual.size(), cudaMemcpyDeviceToHost,
                                  stream.s));
            Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
            Check(cudaStreamSynchronize(stream.s));
            RequireProbe(!fault && actual == expected,
                         "encoded checkpoint differs from byte oracle");
          };
          Check(cudaMemsetAsync(saved.p, 0xaa, saved.bytes, stream.s));
          launch();
          verify();
          ++cases;
          if ((base == 0 || base == 1023 || base == 129024) && extent == tokens) {
            DecodeGraph graph;
            graph.Capture(stream.s, launch);
            for (int replay = 0; replay < 2; ++replay) {
              Check(cudaMemsetAsync(saved.p, 0xbb, saved.bytes, stream.s));
              graph.Launch(stream.s);
              verify();
              ++graphs;
            }
          }
        }
      }
    for (int malformed = 0; malformed < 7; ++malformed) {
      PrefillState state{0, 0, 4, 8192, 0, 1, 1, 1};
      int anchor = 4, blocks = kv_checkpoint::Blocks;
      const void *oldScales = native ? nullptr : old.As<unsigned char>() + layout.scales_offset;
      const void *currentScales = native ? nullptr : current.As<unsigned char>() + 4 * 1024;
      if (malformed == 0)
        anchor = 0;
      if (malformed == 1)
        anchor = 5;
      if (malformed == 2)
        state.valid = 0;
      if (malformed == 3)
        state.capacity = 512;
      if (malformed == 4)
        oldScales = native ? old.p : nullptr;
      if (malformed == 5)
        currentScales = native ? current.p : nullptr;
      if (malformed == 6)
        blocks = 1;
      Check(cudaMemcpyAsync(command.p, &state, sizeof(state), cudaMemcpyHostToDevice, stream.s));
      Check(cudaMemsetAsync(bad.p, 0, 4, stream.s));
      Check(cudaMemsetAsync(saved.p, 0xa5, saved.bytes, stream.s));
      if (native)
        kv_checkpoint::CaptureEncoded<true><<<blocks, kv_checkpoint::Threads, 0, stream.s>>>(
            old.p, old.As<unsigned char>() + layout.v_offset, oldScales, current.p,
            current.As<unsigned char>() + 4 * 544, currentScales, saved.p,
            command.As<PrefillState>(), 4, anchor, bad.As<int>());
      else
        kv_checkpoint::CaptureEncoded<false><<<blocks, kv_checkpoint::Threads, 0, stream.s>>>(
            old.p, old.As<unsigned char>() + layout.v_offset, oldScales, current.p,
            current.As<unsigned char>() + 4 * 512, currentScales, saved.p,
            command.As<PrefillState>(), 4, anchor, bad.As<int>());
      int fault = 0;
      std::vector<unsigned char> actual(saved.bytes);
      Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
      Check(
          cudaMemcpyAsync(actual.data(), saved.p, actual.size(), cudaMemcpyDeviceToHost, stream.s));
      Check(cudaStreamSynchronize(stream.s));
      RequireProbe(fault == 1 &&
                       std::all_of(actual.begin(), actual.end(), [](auto x) { return x == 0xa5; }),
                   "invalid checkpoint command wrote snapshot or escaped guard");
    }
  }
  std::cout << "CHECKPOINT " << cases << " exact byte cases, " << graphs
            << " graph replays and 14 invalid guards passed\n";
}
} // namespace mellum16
