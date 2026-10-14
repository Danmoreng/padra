#include "parser_properties.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("expected corpus directory");
    std::size_t cases = 0;
    for (const auto &entry : std::filesystem::directory_iterator(argv[1])) {
      if (!entry.is_regular_file())
        continue;
      std::ifstream file(entry.path(), std::ios::binary);
      std::string input(std::istreambuf_iterator<char>(file), {});
      mellum16::server::test::CheckParserProperties(input);
      ++cases;
    }
    std::mt19937 random(0x50414452);
    for (int i = 0; i < 512; ++i) {
      std::string input(random() % 512, '\0');
      for (auto &c : input)
        c = static_cast<char>(random() % 256);
      mellum16::server::test::CheckParserProperties(input);
      ++cases;
    }
    for (auto input :
         {std::string(131072, 'x'),
          "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"text\":\"" + std::string(65536, 'x')}) {
      mellum16::server::test::CheckParserProperties(input);
      ++cases;
    }
    std::cout << cases << " deterministic parser property inputs passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
