#include "server/secure_id.h"
#include "server_test_support.h"

void MixedHttpAdmission(text::Tokenizer &tokenizer) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(3);
  auto manual = JsonResponse(
      client.Post("/v1/sessions", R"({"model":"mellum21-exl3-k4-research"})", "application/json"));
  const auto session = manual.find("id")->as_string();
  std::string oldest;
  for (int i = 0; i < 31; ++i) {
    auto response = JsonResponse(client.Post("/v1/responses", ResponseBody, "application/json"));
    if (!i)
      oldest = response.find("id")->as_string();
  }
  auto next = client.Post("/v1/responses", ResponseBody, "application/json");
  Require(next && next->status == 200, "mixed manual/response occupancy rejected evictable root");
  Require(client.Get("/v1/responses/" + oldest)->status == 404, "old response not retired");
  Require(client.Get("/v1/sessions/" + session)->status == 200, "manual session was evicted");
  Require(fixture.coordinator.Sessions().Stats().first == 32, "joint admission exceeded root cap");
  const auto metrics = client.Get("/metrics");
  Require(metrics &&
              metrics->body.find("mellum_response_evicted_records_total 1\n") !=
                  std::string::npos &&
              metrics->body.find("mellum_response_evicted_roots_total 1\n") != std::string::npos,
          "mixed admission eviction counts differ from actual freed resources");
}
std::string Id() {
  return MakeSecureId("resp_").value();
}
struct Stores {
  std::shared_ptr<SessionStore> sessions;
  std::shared_ptr<ResponseStore> responses;
  Stores()
      : sessions(std::make_shared<SessionStore>(SessionLimits{2, 4096, 2048})),
        responses(std::make_shared<ResponseStore>(ResponseStoreLimits{8, 8192, 2048})) {}
  std::shared_ptr<ResponseReservation> Reserve(bool root = true) {
    auto result = responses->Reserve(Id(), "owner", 1024, sessions, root);
    Require(result.ok(), "joint reservation failed");
    return std::move(result).value();
  }
  std::string Keep() {
    auto ticket = Reserve();
    Require(responses->Commit(ticket, "{}", "[]", ticket->root).ok(), "joint commit failed");
    return ticket->id;
  }
};
void SharedRoots() {
  Stores s;
  const auto manual = s.sessions->Create("manual").value();
  auto parent = s.Reserve();
  auto child = s.Reserve(false);
  auto root = parent->root;
  const auto parent_id = parent->id, child_id = child->id;
  Require(s.responses->Commit(parent, "{}", "[]", root).ok() &&
              s.responses->Commit(child, "{}", "[]", root).ok(),
          "shared root setup failed");
  parent.reset();
  child.reset();
  root.reset();
  auto next = s.Reserve();
  Require(!s.responses->Get(parent_id, "owner").ok() && !s.responses->Get(child_id, "owner").ok(),
          "one record incorrectly counted as freed shared root");
  Require(s.responses->Evictions() == std::pair<std::size_t, std::size_t>{2, 1} &&
              s.sessions->Stats().first == 2 && s.sessions->Get(manual.id, "manual").ok(),
          "shared-root eviction count or manual protection broken");
}
void ProtectedRoots() {
  for (bool pinned : {false, true}) {
    Stores s;
    const auto a = s.Keep();
    const auto b = pinned ? s.Keep() : std::string();
    std::shared_ptr<const StoredResponse> pin_a, pin_b;
    std::shared_ptr<SessionLease> lease;
    if (pinned) {
      pin_a = s.responses->Get(a, "owner").value();
      pin_b = s.responses->Get(b, "owner").value();
    } else {
      auto record = s.responses->Get(a, "owner").value();
      lease = s.sessions->Acquire(record->root->session_id, "owner", 128).value();
      Require(s.sessions->Create("manual").ok(), "manual protection setup failed");
    }
    const auto before = s.responses->Stats();
    auto blocked = s.responses->Reserve(Id(), "owner", 1024, s.sessions, true);
    Require(!blocked.ok() && s.responses->Stats() == before &&
                s.responses->Evictions() == std::pair<std::size_t, std::size_t>{0, 0},
            "pinned/active root was evicted or failed admission mutated records");
    pin_a.reset();
    pin_b.reset();
    lease.reset();
    auto admitted = s.Reserve();
    Require(s.sessions->Stats().first == 2, "unprotected root not reusable");
  }
  Stores s;
  auto a = s.Reserve();
  auto b = s.Reserve();
  Require(!s.responses->Reserve(Id(), "owner", 1024, s.sessions, true).ok(),
          "pending roots not protected");
  a.reset();
  b.reset();
  Require(s.responses->Stats().first == 0 && s.sessions->Stats().first == 0,
          "abandoned pending roots leaked");
}
void RollbackAndCallbacks(text::Tokenizer &tokenizer) {
  Stores s;
  for (bool root : {false, true}) {
    try {
      auto ticket = s.Reserve(root);
      throw std::runtime_error("injected failure after reservation");
    } catch (const std::runtime_error &) {
    }
    Require(s.responses->Stats().first == 0 && s.sessions->Stats().first == 0,
            "failed preparation leaked record/root reservation");
  }
  bool callback = false;
  auto ticket = s.responses
                    ->Reserve(Id(), "owner", 1024, s.sessions, true,
                              [&](const std::string &id, const std::string &owner) {
                                // These would deadlock if a root destructor ran under either store
                                // lock.
                                (void)s.responses->Stats();
                                (void)s.sessions->Stats();
                                (void)s.sessions->Delete(id, owner);
                                callback = true;
                              })
                    .value();
  auto id = ticket->id;
  Require(s.responses->Commit(ticket, "{}", "[]", ticket->root).ok(), "callback setup failed");
  ticket.reset();
  Require(s.responses->Delete(id, "owner").ok() && callback && s.sessions->Stats().first == 0,
          "root callback not released outside locks");
  {
    auto throwing = s.responses
                        ->Reserve(Id(), "owner", 1024, s.sessions, true,
                                  [](const std::string &, const std::string &) {
                                    throw std::runtime_error("discard notification");
                                  })
                        .value();
  }
  Require(s.responses->Stats().first == 0 && s.sessions->Stats().first == 0,
          "throwing discard notification prevented root rollback");

  ServerLimits limits;
  limits.queued_bytes = 1;
  Coordinator coordinator(Runtime(std::make_shared<MockState>()), tokenizer, limits);
  {
    auto admission =
        s.responses->Reserve(Id(), "owner", 1024, coordinator.Sessions().shared_from_this(), true)
            .value();
    auto request = Request("injected-submit-failure");
    request.session_id = admission->root->session_id;
    request.session_owner = "owner";
    Require(!coordinator.Submit(std::move(request)).ok(), "submit failure not injected");
  }
  Require(s.responses->Stats().first == 0 && coordinator.Sessions().Stats().first == 0,
          "failed Submit leaked newly admitted root/record");
}
void ExpiryAndBytes() {
  auto now = SessionStore::Clock::now();
  auto sessions = std::make_shared<SessionStore>(SessionLimits{2, 1024, 1024}, [&] { return now; });
  auto responses = std::make_shared<ResponseStore>(ResponseStoreLimits{8, 8192, 2048, 10ms},
                                                   [&] { return now; });
  {
    auto a = responses->Reserve(Id(), "owner", 1024, sessions, true).value();
    auto lease = sessions->Acquire(a->root->session_id, "owner", 1024).value();
    Require(sessions->Commit(lease, "[\"" + std::string(1020, 'x') + "\"]").ok(),
            "byte budget setup failed");
    Require(responses->Commit(a, "{}", "[]", a->root).ok(), "expiry setup failed");
  }
  // Root slots are available, but the full session byte budget needs eviction.
  {
    auto b = responses->Reserve(Id(), "owner", 1024, sessions, true);
    Require(b.ok() && sessions->Stats().second == 2, "joint session byte eviction failed");
    Require(responses->Commit(b.value(), "{}", "[]", b.value()->root).ok(), "expiry commit failed");
  }
  now += 11ms;
  auto c = responses->Reserve(Id(), "owner", 1024, sessions, true);
  Require(c.ok() && responses->Stats().first == 1 && sessions->Stats().first == 1,
          "TTL retirement did not free physical root in same admission");
}
void ConcurrentStores() {
  auto sessions = std::make_shared<SessionStore>(SessionLimits{4, 8192, 2048});
  auto responses = std::make_shared<ResponseStore>(ResponseStoreLimits{4, 8192, 2048});
  std::mutex ids_mutex;
  std::vector<std::string> ids;
  std::vector<std::future<void>> workers;
  for (int thread = 0; thread < 4; ++thread)
    workers.push_back(std::async(std::launch::async, [&, thread] {
      for (int i = 0; i < 100; ++i) {
        if (thread % 2) {
          auto manual = sessions->Create("manual");
          if (manual.ok())
            Require(sessions->Delete(manual.value().id, "manual").ok(),
                    "concurrent manual deletion failed");
        } else {
          auto ticket = responses->Reserve(Id(), "owner", 1024, sessions, true);
          if (ticket.ok()) {
            Require(responses->Commit(ticket.value(), "{}", "[]", ticket.value()->root).ok(),
                    "concurrent response commit failed");
            std::lock_guard lock(ids_mutex);
            ids.push_back(ticket.value()->id);
          } else
            Require(ticket.status().code() == StatusCode::kResourceExhausted,
                    "unexpected concurrent admission error");
        }
        Require(sessions->Stats().first <= 4 && responses->Stats().first <= 4 &&
                    responses->Stats().second <= 8192,
                "concurrent admission exceeded limits");
      }
    }));
  for (auto &worker : workers)
    worker.get();
  for (const auto &id : ids)
    (void)responses->Delete(id, "owner");
  Require(responses->Stats().first == 0 && sessions->Stats().first == 0,
          "concurrent admissions leaked roots");
}
void ProtectedHttp(text::Tokenizer &tokenizer) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  httplib::Client client("127.0.0.1", fixture.port);
  for (int i = 0; i < 32; ++i)
    (void)JsonResponse(client.Post("/v1/sessions", R"({"model":"mellum21-exl3-k4-research"})",
                                   "application/json"));
  auto blocked = client.Post("/v1/responses", ResponseBody, "application/json");
  Require(blocked && blocked->status == 429 && fixture.coordinator.Sessions().Stats().first == 32,
          "protected manual session capacity was evicted/increased");
}
int main() {
  try {
    auto tokenizer = text::Tokenizer::Load({}, {});
    MixedHttpAdmission(*tokenizer.value());
    SharedRoots();
    ProtectedRoots();
    RollbackAndCallbacks(*tokenizer.value());
    ExpiryAndBytes();
    ConcurrentStores();
    ProtectedHttp(*tokenizer.value());
    std::cout << "PASS joint response/root admission\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
