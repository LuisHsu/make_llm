// nn.hpp 的比對測試：把 golden 檔裡的權重寫進模型，再和 tools/gen_nn_golden.py
// 以 PyTorch float64 產生的輸出、loss 與參數梯度比對。
//
// 介面約定（include/nn.hpp，namespace mg）：
//   - mg::Neuron n{nin};                 nin 個輸入的單一 neuron
//   - mg::Layer  l{nin, nout};           nout 個 neuron
//   - mg::MLP    m{nin, {4, 4, 1}};      nouts 為各層輸出數
//   - 大小一律 size_t；MLP 的 nouts 型別為 std::vector<size_t>
//   - 呼叫：n(xs) 回傳 mg::Value；l(xs)、m(xs) 回傳 std::vector<mg::Value>
//     （nout == 1 時也是長度 1 的 vector，和 Python 版回傳單一 Value 的寫法不同）
//     參數 xs 型別為 const std::vector<mg::Value>&
//   - parameters() 回傳 std::vector<mg::Value>，順序是
//     層 → neuron → 該 neuron 的權重（依輸入順序）→ bias
//   - 每個 neuron 都套 tanh（和 01_micrograd/nn.py 一樣）；換成別的非線性函數
//     golden 值就對不上，要重新產生
//   - parameters() 回傳的 Value 與模型內部共用節點，所以測試直接寫 p.data()、
//     p.grad() 就會改到真正的參數
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "engine.hpp"
#include "nn.hpp"

namespace {

// 型別不同（Neuron 回傳單一 Value），用這層包裝統一成同一種呼叫方式
struct Model {
  std::function<std::vector<mg::Value>(const std::vector<mg::Value>&)> forward;
  std::function<std::vector<mg::Value>()> parameters;
};

Model MakeModel(const std::string& kind, size_t nin, const std::vector<size_t>& nouts) {
  if (kind == "neuron") {
    auto n = std::make_shared<mg::Neuron>(nin);
    return Model{[n](const std::vector<mg::Value>& xs) {
                   return std::vector<mg::Value>{(*n)(xs)};
                 },
                 [n] { return n->parameters(); }};
  }
  if (kind == "layer") {
    auto l = std::make_shared<mg::Layer>(nin, nouts.at(0));
    return Model{[l](const std::vector<mg::Value>& xs) { return (*l)(xs); },
                 [l] { return l->parameters(); }};
  }
  auto m = std::make_shared<mg::MLP>(nin, nouts);
  return Model{[m](const std::vector<mg::Value>& xs) { return (*m)(xs); },
               [m] { return m->parameters(); }};
}

double Tol(double want) { return 1e-9 * std::max(1.0, std::abs(want)); }

std::vector<mg::Value> AsValues(const std::vector<double>& xs) {
  std::vector<mg::Value> vs;
  vs.reserve(xs.size());
  for (const double x : xs) {
    vs.emplace_back(x);
  }
  return vs;
}

nlohmann::json LoadGolden() {
  const std::filesystem::path path = std::filesystem::path(GOLDEN_DIR) / "nn_cases.json";
  EXPECT_TRUE(std::filesystem::exists(path))
      << "找不到 " << path
      << "，請先執行 uv run python 01_micrograd/cpp/tools/gen_nn_golden.py";
  std::ifstream in(path);
  nlohmann::json j;
  in >> j;
  return j;
}

class NnCase : public testing::TestWithParam<nlohmann::json> {};

TEST_P(NnCase, MatchesPyTorch) {
  const nlohmann::json& c = GetParam();
  const auto name = c.at("name").get<std::string>();
  const auto kind = c.at("kind").get<std::string>();
  const auto nin = c.at("nin").get<size_t>();
  const auto nouts = c.at("nouts").get<std::vector<size_t>>();
  const auto want_params = c.at("params").get<std::vector<double>>();
  const auto inputs = c.at("inputs").get<std::vector<std::vector<double>>>();
  const auto targets = c.at("targets").get<std::vector<std::vector<double>>>();
  const auto want_outputs = c.at("outputs").get<std::vector<std::vector<double>>>();
  const auto want_grads = c.at("param_grads").get<std::vector<double>>();
  const double want_loss = c.at("loss").get<double>();

  Model model = MakeModel(kind, nin, nouts);
  std::vector<mg::Value> ps = model.parameters();
  ASSERT_EQ(ps.size(), want_params.size())
      << name << "：參數個數不對，每個 neuron 應該是 nin 個權重加 1 個 bias";
  for (size_t i = 0; i < ps.size(); ++i) {
    ps[i].data() = want_params[i];
  }

  mg::Value loss{0.0};
  for (size_t s = 0; s < inputs.size(); ++s) {
    std::vector<mg::Value> xs = AsValues(inputs[s]);
    std::vector<mg::Value> out = model.forward(xs);
    ASSERT_EQ(out.size(), targets[s].size()) << name << "：第 " << s << " 筆的輸出個數不對";
    for (size_t j = 0; j < out.size(); ++j) {
      EXPECT_NEAR(out[j].data(), want_outputs[s][j], Tol(want_outputs[s][j]))
          << name << "：第 " << s << " 筆的第 " << j << " 個輸出";
      loss = loss + (out[j] - targets[s][j]).pow(2.0);
    }
  }

  EXPECT_NEAR(loss.data(), want_loss, Tol(want_loss)) << name << " 的 loss";

  loss.backward();
  ASSERT_EQ(ps.size(), want_grads.size()) << name;
  for (size_t i = 0; i < ps.size(); ++i) {
    EXPECT_NEAR(ps[i].grad(), want_grads[i], Tol(want_grads[i]))
        << name << " 的第 " << i << " 個參數（順序：層 → neuron → 權重 → bias）";
  }
}

INSTANTIATE_TEST_SUITE_P(
    Golden, NnCase, testing::ValuesIn([] {
      // 同 test_engine.cpp：LoadGolden() 回傳暫時物件，先綁到具名變數再走訪
      const nlohmann::json golden = LoadGolden();
      std::vector<nlohmann::json> cases;
      for (const auto& c : golden.at("cases")) {
        cases.push_back(c);
      }
      return cases;
    }()),
    [](const testing::TestParamInfo<nlohmann::json>& info) {
      return info.param.at("name").get<std::string>();
    });

TEST(NnShape, NeuronHasWeightsPlusBias) {
  mg::Neuron n{3};
  EXPECT_EQ(n.parameters().size(), 4u);
  mg::Value out = n(AsValues({1.0, -2.0, 0.5}));
  EXPECT_TRUE(std::isfinite(out.data()));
}

TEST(NnShape, LayerHasOneNeuronPerOutput) {
  mg::Layer l{3, 4};
  EXPECT_EQ(l.parameters().size(), 4u * (3u + 1u));
  EXPECT_EQ(l(AsValues({1.0, -2.0, 0.5})).size(), 4u);
}

// nout == 1 時仍然是長度 1 的 vector（Python 版在這裡回傳單一 Value，C++ 版不跟）
TEST(NnShape, LayerWithSingleOutputStillReturnsVector) {
  mg::Layer l{3, 1};
  EXPECT_EQ(l(AsValues({1.0, -2.0, 0.5})).size(), 1u);
}

TEST(NnShape, MlpCountsEveryLayer) {
  mg::MLP m{3, std::vector<size_t>{4, 4, 1}};
  EXPECT_EQ(m.parameters().size(), 4u * 4u + 4u * 5u + 1u * 5u);
  EXPECT_EQ(m(AsValues({1.0, -2.0, 0.5})).size(), 1u);
}

// 參數必須是各自獨立的節點，而且 parameters() 每次回傳的順序要一樣
TEST(NnShape, ParametersAreDistinctAndStablyOrdered) {
  mg::MLP m{2, std::vector<size_t>{3, 2}};
  std::vector<mg::Value> ps = m.parameters();
  for (size_t i = 0; i < ps.size(); ++i) {
    ps[i].data() = static_cast<double>(i);
  }
  std::vector<mg::Value> again = m.parameters();
  ASSERT_EQ(again.size(), ps.size());
  for (size_t i = 0; i < again.size(); ++i) {
    EXPECT_DOUBLE_EQ(again[i].data(), static_cast<double>(i))
        << "第 " << i << " 個參數不是獨立節點，或兩次 parameters() 的順序不同";
  }
}

// 權重全部相同（例如忘了隨機初始化）時，同一層的 neuron 會一直保持一模一樣
TEST(NnInit, WeightsAreNotAllIdentical) {
  mg::Neuron n{16};
  std::vector<mg::Value> ps = n.parameters();
  ASSERT_EQ(ps.size(), 17u);
  bool all_same = true;
  for (size_t i = 1; i < ps.size(); ++i) {
    if (ps[i].data() != ps[0].data()) {
      all_same = false;
      break;
    }
  }
  EXPECT_FALSE(all_same) << "17 個參數的初始值完全相同，看起來沒有隨機初始化";
}

// 和 Python 版 test_micrograd_nn.py 的 test_training_reduces_loss 對應
TEST(NnTraining, ReducesLoss) {
  const std::vector<std::vector<double>> xs = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  const std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};

  mg::MLP model{3, std::vector<size_t>{4, 4, 1}};
  std::vector<mg::Value> ps = model.parameters();

  const auto loss_fn = [&] {
    mg::Value loss{0.0};
    for (size_t i = 0; i < xs.size(); ++i) {
      std::vector<mg::Value> out = model(AsValues(xs[i]));
      EXPECT_EQ(out.size(), 1u);
      loss = loss + (out[0] - ys[i]).pow(2.0);
    }
    return loss;
  };

  const double first = loss_fn().data();
  for (int step = 0; step < 50; ++step) {
    for (auto& p : ps) {
      p.grad() = 0.0;
    }
    mg::Value loss = loss_fn();
    loss.backward();
    for (auto& p : ps) {
      p.data() -= 0.01 * p.grad();
    }
  }
  const double last = loss_fn().data();

  EXPECT_LT(last, first) << "loss 沒有下降：" << first << " -> " << last;
}

}  // namespace
