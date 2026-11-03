#include "runtime/shared_weight_registry.h"
#include <stdexcept>
#include <thread>
#include <vector>
void Require(bool condition) {
  if (!condition)
    throw std::runtime_error("registry test failed");
}
#include <memory>
using namespace mellum16;
int main() {
  SharedWeightRegistry<int> registry;
  SharedWeightKey key{0, "manifest", "weights", "head", "attention", false};
  auto lock = registry.Acquire();
  auto first = std::make_shared<int>(42);
  registry.Publish(key, first, lock);
  auto second = registry.Find(key, lock);
  Require(first == second && *second == 42);
  Require(!registry.Find({1, "manifest", "weights", "head", "attention", false}, lock));
  Require(!registry.Find({0, "other", "weights", "head", "attention", false}, lock));
  Require(!registry.Find({0, "manifest", "other", "head", "attention", false}, lock));
  Require(!registry.Find({0, "manifest", "weights", "head", "attention", true}, lock));
  Require(!registry.Find({0, "manifest", "weights", "other", "attention", false}, lock));
  Require(!registry.Find({0, "manifest", "weights", "head", "other", false}, lock));
  first.reset();
  Require(bool(registry.Find(key, lock)));
  second.reset();
  Require(!registry.Find(key, lock));
  SharedWeightRegistry<int, SharedDraftWeightKey> drafts;
  auto draftLock = drafts.Acquire();
  SharedDraftWeightKey draftKey{0, "draft-manifest", "draft-weights", false, true, true};
  auto draft = std::make_shared<int>(17);
  drafts.Publish(draftKey, draft, draftLock);
  Require(drafts.Find(draftKey, draftLock) == draft);
  Require(!drafts.Find({1, "draft-manifest", "draft-weights", false, true, true}, draftLock));
  Require(!drafts.Find({0, "different", "draft-weights", false, true, true}, draftLock));
  Require(!drafts.Find({0, "draft-manifest", "different", false, true, true}, draftLock));
  Require(!drafts.Find({0, "draft-manifest", "draft-weights", true, true, true}, draftLock));
  Require(!drafts.Find({0, "draft-manifest", "draft-weights", false, false, true}, draftLock));
  Require(!drafts.Find({0, "draft-manifest", "draft-weights", false, true, false}, draftLock));
  draft.reset();
  Require(!drafts.Find(draftKey, draftLock));
  draftLock.unlock();
  std::vector<std::shared_ptr<int>> owners(8);
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < owners.size(); ++i) {
    threads.emplace_back([&, i] {
      auto held = drafts.Acquire();
      auto value = drafts.Find(draftKey, held);
      if (!value) {
        value = std::make_shared<int>(23);
        drafts.Publish(draftKey, value, held);
      }
      owners[i] = std::move(value);
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &owner : owners)
    Require(owner == owners.front() && *owner == 23);

  Require(NewRuntimeAllocationBytes(1000, 600, 200) == 200);
  Require(NewRuntimeAllocationBytes(1000, 0, 0) == 1000);
  bool rejected = false;
  try {
    (void)NewRuntimeAllocationBytes(1000, 600, 401);
  } catch (const std::overflow_error &) {
    rejected = true;
  }
  Require(rejected);
}
