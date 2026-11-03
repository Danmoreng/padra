#include "runtime/shared_weight_registry.h"
#include <stdexcept>
void Require(bool condition) {
  if (!condition)
    throw std::runtime_error("registry test failed");
}
#include <memory>
using namespace mellum16;
int main() {
  SharedWeightRegistry<int> registry;
  SharedWeightKey key{0, "manifest", "weights", false};
  auto lock = registry.Acquire();
  auto first = std::make_shared<int>(42);
  registry.Publish(key, first, lock);
  auto second = registry.Find(key, lock);
  Require(first == second && *second == 42);
  Require(!registry.Find({1, "manifest", "weights", false}, lock));
  Require(!registry.Find({0, "other", "weights", false}, lock));
  Require(!registry.Find({0, "manifest", "other", false}, lock));
  Require(!registry.Find({0, "manifest", "weights", true}, lock));
  first.reset();
  Require(bool(registry.Find(key, lock)));
  second.reset();
  Require(!registry.Find(key, lock));
}
