#pragma once
// Included inside the SIMT operator fixture's namespace.
void Q8PairedAttention(bool quick = false) {
  constexpr int capacity = 8192;
  Stream stream;
  Buffer query(2 * 4096 * 4), keyInput(2 * 512 * 4), valueInput(2 * 512 * 4),
      encoded(2 * 128 * sizeof(q8_native::QueryBlock)), states(2 * sizeof(DecodeState)),
      batch(sizeof(SimtBatch)), bad(4), saved(2 * sizeof(q8_native::CacheRow)),
      chunk(4 * sizeof(q8_native::CacheRow)), scratch(2 * 32 * 32 * 129 * 4), output(2 * 4096 * 4),
      reference(output.bytes);
  std::vector<float> q(8192), k(1024), v(1024), expected(8192), actual(8192);
  for (int i = 0; i < 8192; ++i)
    q[i] = BHost(std::sin(float(i) * .11f));
  for (int i = 0; i < 1024; ++i) {
    k[i] = BHost(std::cos(float(i) * .17f));
    v[i] = BHost(std::sin(float(i) * .13f));
  }
  Check(cudaMemcpy(query.p, q.data(), query.bytes, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(keyInput.p, k.data(), keyInput.bytes, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(valueInput.p, v.data(), valueInput.bytes, cudaMemcpyHostToDevice));
  int cases = 0;
  for (bool local : {false, true}) {
    const int cap = local ? 1024 : capacity, splits = local ? 4 : 32;
    Buffer keys(cap * sizeof(q8_native::CacheRow)), values(keys.bytes);
    std::vector<q8_native::CacheRow> historyK(cap), historyV(cap);
    for (int i = 0; i < cap; ++i) {
      for (int d = 0; d < 16; ++d) {
        historyK[i].d[d] = __float2half_rn(float(1 + (i + d * 7) % 31) / 512.f);
        historyV[i].d[d] = __float2half_rn(float(1 + (i + d * 11) % 31) / 256.f);
      }
      for (int d = 0; d < 512; ++d) {
        historyK[i].qs[d] = static_cast<signed char>((i * 13 + d * 19) % 255 - 127);
        historyV[i].qs[d] = static_cast<signed char>((i * 17 + d * 23) % 255 - 127);
      }
    }
    for (int position : {0, 1, 126, 127, 128, 1022, 1023, 1024, 2047, 4095, 8190}) {
      if (quick && position != 127 && position != 1023 && position != 8190)
        continue;
      const SimtBatch shape{position, 2, capacity};
      Check(cudaMemcpy(batch.p, &shape, sizeof(shape), cudaMemcpyHostToDevice));
      Check(cudaMemset(bad.p, 0, 4));
      q8_simt::PairStates<<<1, 2, 0, stream.s>>>(batch.As<SimtBatch>(), states.As<DecodeState>(),
                                                 bad.As<int>());
      const auto reset = [&] {
        Check(
            cudaMemcpyAsync(keys.p, historyK.data(), keys.bytes, cudaMemcpyHostToDevice, stream.s));
        Check(cudaMemcpyAsync(values.p, historyV.data(), values.bytes, cudaMemcpyHostToDevice,
                              stream.s));
      };
      reset();
      const q8_native::CacheRow old[2] = {historyK[(position + 1) % cap],
                                          historyV[(position + 1) % cap]};
      Check(cudaMemcpyAsync(saved.p, old, sizeof(old), cudaMemcpyHostToDevice, stream.s));
      for (int row = 0; row < 2; ++row) {
        q8_native::EncodeKV<true><<<1, 128, 0, stream.s>>>(
            keyInput.As<float>() + row * 512, valueInput.As<float>() + row * 512,
            keys.As<q8_native::CacheRow>(), values.As<q8_native::CacheRow>(), 1, bad.As<int>(),
            states.As<DecodeState>() + row, cap, capacity);
        q8_native::EncodeQuery<><<<8, 128, 0, stream.s>>>(
            query.As<float>() + row * 4096, encoded.As<q8_native::QueryBlock>(), bad.As<int>());
        auto *lse = scratch.As<float>() + splits * 32 * 128;
        if (local) {
          q8_native::Split<true><<<dim3(8, splits), 128, 0, stream.s>>>(
              encoded.As<q8_native::QueryBlock>(), keys.As<q8_native::CacheRow>(),
              values.As<q8_native::CacheRow>(), scratch.As<float>(), lse,
              states.As<DecodeState>() + row, cap, splits, bad.As<int>(), capacity);
          q8_native::Merge<true, float><<<8, 128, 0, stream.s>>>(
              scratch.As<float>(), lse, reference.As<float>() + row * 4096,
              states.As<DecodeState>() + row, splits, bad.As<int>(), capacity);
        } else {
          q8_native::Split<false><<<dim3(8, splits), 128, 0, stream.s>>>(
              encoded.As<q8_native::QueryBlock>(), keys.As<q8_native::CacheRow>(),
              values.As<q8_native::CacheRow>(), scratch.As<float>(), lse,
              states.As<DecodeState>() + row, cap, splits, bad.As<int>(), capacity);
          q8_native::Merge<false, float><<<8, 128, 0, stream.s>>>(
              scratch.As<float>(), lse, reference.As<float>() + row * 4096,
              states.As<DecodeState>() + row, splits, bad.As<int>(), capacity);
        }
      }
      std::vector<unsigned char> expectedK(keys.bytes), expectedV(values.bytes), check(keys.bytes);
      Check(cudaMemcpyAsync(expected.data(), reference.p, reference.bytes, cudaMemcpyDeviceToHost,
                            stream.s));
      Check(
          cudaMemcpyAsync(expectedK.data(), keys.p, keys.bytes, cudaMemcpyDeviceToHost, stream.s));
      Check(cudaMemcpyAsync(expectedV.data(), values.p, values.bytes, cudaMemcpyDeviceToHost,
                            stream.s));
      Check(cudaStreamSynchronize(stream.s));
      reset();
      const auto launch = [&] {
        q8_simt::PairStates<<<1, 2, 0, stream.s>>>(batch.As<SimtBatch>(), states.As<DecodeState>(),
                                                   bad.As<int>());
        q8_native::EncodeKV<><<<2, 128, 0, stream.s>>>(
            keyInput.As<float>(), valueInput.As<float>(), chunk.As<q8_native::CacheRow>(),
            chunk.As<q8_native::CacheRow>() + 2, 2, bad.As<int>(), nullptr, 0, capacity,
            &batch.As<SimtBatch>()->rows);
        q8_simt::Commit<<<2, 128, 0, stream.s>>>(
            chunk.As<q8_native::CacheRow>(), chunk.As<q8_native::CacheRow>() + 2,
            keys.As<q8_native::CacheRow>(), values.As<q8_native::CacheRow>(), batch.As<SimtBatch>(),
            cap);
        q8_simt::AttentionPair(query.As<float>(), keys.As<q8_native::CacheRow>(),
                               values.As<q8_native::CacheRow>(), output.As<float>(),
                               encoded.As<q8_native::QueryBlock>(), states.As<DecodeState>(), cap,
                               local, scratch.As<float>(), saved.As<q8_native::CacheRow>(),
                               bad.As<int>(), stream.s, capacity);
      };
      const auto verify = [&] {
        int fault = 0;
        Check(cudaMemcpyAsync(actual.data(), output.p, output.bytes, cudaMemcpyDeviceToHost,
                              stream.s));
        Check(cudaMemcpyAsync(&fault, bad.p, 4, cudaMemcpyDeviceToHost, stream.s));
        Check(cudaStreamSynchronize(stream.s));
        Require(!fault && !std::memcmp(actual.data(), expected.data(), output.bytes),
                "paired Q8 attention differs from sequential values");
        Check(cudaMemcpy(check.data(), keys.p, keys.bytes, cudaMemcpyDeviceToHost));
        Require(check == expectedK, "paired Q8 key commit differs from sequential cache");
        Check(cudaMemcpy(check.data(), values.p, values.bytes, cudaMemcpyDeviceToHost));
        Require(check == expectedV, "paired Q8 value commit differs from sequential cache");
      };
      Check(cudaMemsetAsync(scratch.p, 0xff, scratch.bytes, stream.s));
      launch();
      verify();
      DecodeGraph graph;
      graph.Capture(stream.s, launch);
      for (int replay = 0; replay < 2; ++replay) {
        Check(cudaMemsetAsync(scratch.p, 0xff, scratch.bytes, stream.s));
        graph.Launch(stream.s);
        verify();
      }
      ++cases;
    }
  }
  for (const SimtBatch invalid :
       {SimtBatch{8191, 2, 8192}, SimtBatch{0, 1, 8192}, SimtBatch{-1, 2, 8192}}) {
    Check(cudaMemset(bad.p, 0, 4));
    Check(cudaMemcpy(batch.p, &invalid, sizeof(invalid), cudaMemcpyHostToDevice));
    q8_simt::PairStates<<<1, 2>>>(batch.As<SimtBatch>(), states.As<DecodeState>(), bad.As<int>());
    int fault = 0;
    Check(cudaMemcpy(&fault, bad.p, 4, cudaMemcpyDeviceToHost));
    Require(fault == 1, "paired Q8 states accepted an invalid batch");
  }
  std::cout << "Paired Q8 attention: " << cases << " exact cases, " << cases * 2
            << " graph replays and invalid-state checks passed\n";
}
