#include "text/tokenizer.h"
namespace mellum16::text {
StreamDecoder::StreamDecoder(const Tokenizer &tokenizer, bool skip)
    : tokenizer_(tokenizer), skip_special_(skip) {
  pending_.reserve(tokenizer.MaxRawTokenBytes() + 3);
  output_.reserve((tokenizer.MaxRawTokenBytes() + 3) * 3);
}
std::string_view StreamDecoder::Push(int id) {
  pending_.append(tokenizer_.RawToken(id, skip_special_));
  return Process(false);
}
std::string_view StreamDecoder::Finish() {
  return Process(true);
}
std::string_view StreamDecoder::Process(bool final) {
  output_.clear();
  std::size_t i = 0;
  while (i < pending_.size()) {
    const auto lead = static_cast<unsigned char>(pending_[i]);
    int length = lead < 128                     ? 1
                 : lead >= 0xc2 && lead <= 0xdf ? 2
                 : lead >= 0xe0 && lead <= 0xef ? 3
                 : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                : 0;
    if (length == 1) {
      output_.push_back(pending_[i++]);
      continue;
    }
    if (length == 0) {
      output_.append("\xef\xbf\xbd");
      ++i;
      continue;
    }
    int good = 1;
    bool invalid = false;
    for (int j = 1; j < length && i + static_cast<std::size_t>(j) < pending_.size(); ++j) {
      const auto c = static_cast<unsigned char>(pending_[i + j]);
      bool valid = c >= 128 && c <= 191;
      if (j == 1) {
        if (lead == 0xe0)
          valid = valid && c >= 0xa0;
        if (lead == 0xed)
          valid = valid && c < 0xa0;
        if (lead == 0xf0)
          valid = valid && c >= 0x90;
        if (lead == 0xf4)
          valid = valid && c < 0x90;
      }
      if (!valid) {
        invalid = true;
        break;
      }
      ++good;
    }
    if (invalid) {
      output_.append("\xef\xbf\xbd");
      i += good;
      continue;
    }
    if (good < length) {
      if (final) {
        output_.append("\xef\xbf\xbd");
        i += good;
      }
      break;
    }
    output_.append(pending_, i, length);
    i += length;
  }
  pending_.erase(0, i);
  return output_;
}
} // namespace mellum16::text
