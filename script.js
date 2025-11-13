// ====== 全域狀態 ======
let data = [];
let predictions = {};
let modelMetrics = {};
let selectedCity = "台北市";
let forecastQuarters = 8;
let loading = true;

// Chart.js 實例
let trendChart = null;
let featureChart = null;
let treeChart = null;

// 城市與欄位對應
const cities = ["台北市", "新北市", "桃園市", "台中市", "台南市", "高雄市"];

const cityMapping = {
  台北市: "台北市_單坪價格",
  新北市: "新北市_單坪價格",
  桃園市: "桃園市_單坪價格",
  台中市: "台中市_單坪價格",
  台南市: "台南市_單坪價格",
  高雄市: "高雄市_單坪價格",
};

const inventoryMapping = {
  台北市: "餘屋_台北市",
  新北市: "餘屋_新北市",
  桃園市: "餘屋_桃園市",
  台中市: "餘屋_台中市",
  台南市: "餘屋_台南市",
  高雄市: "餘屋_高雄市",
};

// ====== 決策樹節點 ======
class DecisionTreeNode {
  constructor(depth = 0, maxDepth = 3) {
    this.depth = depth;
    this.maxDepth = maxDepth;
    this.featureIndex = null;
    this.threshold = null;
    this.leftChild = null;
    this.rightChild = null;
    this.value = null;
    this.isLeaf = false;
  }

  // 計算 MSE
  calculateMSE(y) {
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    return y.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / y.length;
  }

  // 尋找最佳分割
  findBestSplit(X, y) {
    const nFeatures = X[0].length;
    let bestMSE = Infinity;
    let bestFeature = null;
    let bestThreshold = null;

    for (let featureIdx = 0; featureIdx < nFeatures; featureIdx++) {
      const values = X.map((row) => row[featureIdx]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;

        const leftIndices = [];
        const rightIndices = [];

        X.forEach((row, idx) => {
          if (row[featureIdx] <= threshold) {
            leftIndices.push(idx);
          } else {
            rightIndices.push(idx);
          }
        });

        if (leftIndices.length === 0 || rightIndices.length === 0) continue;

        const leftY = leftIndices.map((idx) => y[idx]);
        const rightY = rightIndices.map((idx) => y[idx]);

        const leftMSE = this.calculateMSE(leftY);
        const rightMSE = this.calculateMSE(rightY);
        const weightedMSE =
          (leftY.length * leftMSE + rightY.length * rightMSE) / y.length;

        if (weightedMSE < bestMSE) {
          bestMSE = weightedMSE;
          bestFeature = featureIdx;
          bestThreshold = threshold;
        }
      }
    }

    return { featureIdx: bestFeature, threshold: bestThreshold, mse: bestMSE };
  }

  // 訓練決策樹
  fit(X, y) {
    if (this.depth >= this.maxDepth || y.length < 5) {
      this.isLeaf = true;
      this.value = y.reduce((a, b) => a + b, 0) / y.length;
      return;
    }

    const { featureIdx, threshold } = this.findBestSplit(X, y);

    if (featureIdx === null) {
      this.isLeaf = true;
      this.value = y.reduce((a, b) => a + b, 0) / y.length;
      return;
    }

    this.featureIndex = featureIdx;
    this.threshold = threshold;

    const leftX = [];
    const leftY = [];
    const rightX = [];
    const rightY = [];

    X.forEach((row, idx) => {
      if (row[featureIdx] <= threshold) {
        leftX.push(row);
        leftY.push(y[idx]);
      } else {
        rightX.push(row);
        rightY.push(y[idx]);
      }
    });

    if (leftY.length > 0) {
      this.leftChild = new DecisionTreeNode(this.depth + 1, this.maxDepth);
      this.leftChild.fit(leftX, leftY);
    }

    if (rightY.length > 0) {
      this.rightChild = new DecisionTreeNode(this.depth + 1, this.maxDepth);
      this.rightChild.fit(rightX, rightY);
    }
  }

  // 預測單筆
  predict(x) {
    if (this.isLeaf) {
      return this.value;
    }

    if (x[this.featureIndex] <= this.threshold) {
      return this.leftChild ? this.leftChild.predict(x) : this.value || 0;
    } else {
      return this.rightChild ? this.rightChild.predict(x) : this.value || 0;
    }
  }
}

// ====== 梯度提升模型 ======
class GradientBoostingRegressor {
  constructor(nEstimators = 50, learningRate = 0.1, maxDepth = 3) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
    this.trees = [];
    this.initialPrediction = 0;
    this.featureNames = [];
    this.r2Score = 0;
    this.mae = 0;
    this.treeContributions = [];
  }

  fit(X, y, featureNames) {
    this.featureNames = featureNames;
    this.initialPrediction = y.reduce((a, b) => a + b, 0) / y.length;

    let currentPredictions = new Array(y.length).fill(this.initialPrediction);

    for (let i = 0; i < this.nEstimators; i++) {
      const residuals = y.map((val, idx) => val - currentPredictions[idx]);

      const tree = new DecisionTreeNode(0, this.maxDepth);
      tree.fit(X, residuals);

      const treePredictions = X.map((x) => tree.predict(x));
      currentPredictions = currentPredictions.map(
        (pred, idx) => pred + this.learningRate * treePredictions[idx]
      );

      this.trees.push(tree);

      const contribution =
        residuals.reduce((sum, r) => sum + Math.abs(r), 0) / residuals.length;
      this.treeContributions.push(contribution);
    }

    this.calculateMetrics(X, y);
  }

  predict(X) {
    if (Array.isArray(X[0])) {
      return X.map((x) => this.predictSingle(x));
    } else {
      return this.predictSingle(X);
    }
  }

  predictSingle(x) {
    let prediction = this.initialPrediction;
    for (const tree of this.trees) {
      prediction += this.learningRate * tree.predict(x);
    }
    return prediction;
  }

  calculateMetrics(X, y) {
    const predictions = this.predict(X);
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;

    const ssTotal = y.reduce(
      (sum, val) => sum + Math.pow(val - yMean, 2),
      0
    );
    const ssResidual = y.reduce(
      (sum, val, i) => sum + Math.pow(val - predictions[i], 2),
      0
    );
    this.r2Score = 1 - ssResidual / ssTotal;

    this.mae =
      y.reduce((sum, val, i) => sum + Math.abs(val - predictions[i]), 0) /
      y.length;
  }

  getFeatureImportance() {
    const importance = new Array(this.featureNames.length).fill(0);

    const countFeatureUsage = (node) => {
      if (!node || node.isLeaf) return;
      if (node.featureIndex !== null) importance[node.featureIndex]++;
      if (node.leftChild) countFeatureUsage(node.leftChild);
      if (node.rightChild) countFeatureUsage(node.rightChild);
    };

    this.trees.forEach((tree) => countFeatureUsage(tree));

    const total = importance.reduce((a, b) => a + b, 0);
    return this.featureNames
      .map((name, i) => ({
        feature: name,
        importance: total > 0 ? (importance[i] / total) * 100 : 0,
      }))
      .sort((a, b) => b.importance - a.importance);
  }

  getTreeContributions() {
    return this.treeContributions.map((c, i) => ({
      tree: i + 1,
      contribution: c,
    }));
  }
}

// ====== 資料相關 ======

function calculateConfidence(stepsAhead, r2Score) {
  const baseConfidence = r2Score * 100;
  return Math.max(50, baseConfidence - stepsAhead * 3);
}

function getChartData() {
  if (!data.length || !predictions[selectedCity]) return [];
  const priceColumn = cityMapping[selectedCity];

  const historical = data.map((row) => ({
    period: row.period,
    actual: row[priceColumn],
    type: "historical",
  }));

  const forecast = predictions[selectedCity].map((p) => ({
    period: p.period,
    predicted: p.predictedPrice,
    upper: p.predictedPrice * (1 + (100 - p.confidence) / 200),
    lower: p.predictedPrice * (1 - (100 - p.confidence) / 200),
    type: "forecast",
  }));

  return [...historical, ...forecast];
}

function getLatestPrice() {
  if (!data.length) return null;
  const priceColumn = cityMapping[selectedCity];
  return data[data.length - 1][priceColumn];
}

function getPredictedChange() {
  if (!predictions[selectedCity] || !data.length) return null;
  const priceColumn = cityMapping[selectedCity];
  const currentPrice = data[data.length - 1][priceColumn];
  const futurePrice =
    predictions[selectedCity][forecastQuarters - 1].predictedPrice;
  const change = (((futurePrice - currentPrice) / currentPrice) * 100).toFixed(
    2
  );
  return { change, futurePrice };
}

// ====== 訓練 + 預測 ======

function trainAndPredict(historicalData) {
  const newPredictions = {};
  const metrics = {};

  cities.forEach((city) => {
    const priceColumn = cityMapping[city];
    const inventoryColumn = inventoryMapping[city];

    const trainingData = historicalData
      .map((row) => ({
        price: row[priceColumn],
        inventory: row[inventoryColumn],
        rate: row["重貼現率_pct"],
        mortgage: row["房貸餘額_季末"],
        m2: row.M2,
        avgPrice: row["電梯大廈_六都均價"],
        year: row.year,
        quarter: row.quarter,
      }))
      .filter((d) => d.price != null && d.inventory != null);

    const X = trainingData.map((d) => [
      d.rate,
      d.mortgage / 1_000_000,
      d.m2 / 10_000,
      d.inventory / 1_000,
      d.avgPrice,
      d.year - 2010,
      Math.sin((d.quarter * Math.PI) / 2),
      Math.cos((d.quarter * Math.PI) / 2),
    ]);

    const y = trainingData.map((d) => d.price);

    const featureNames = [
      "重貼現率",
      "房貸餘額",
      "M2貨幣供給",
      "餘屋數",
      "六都均價",
      "時間趨勢",
      "季節性(sin)",
      "季節性(cos)",
    ];

    const model = new GradientBoostingRegressor(50, 0.1, 3);
    model.fit(X, y, featureNames);

    metrics[city] = {
      r2Score: model.r2Score,
      mae: model.mae,
      featureImportance: model.getFeatureImportance(),
      treeContributions: model.getTreeContributions().slice(0, 10),
      nTrees: model.trees.length,
    };

    const lastData = historicalData[historicalData.length - 1];
    const futurePoints = [];

    for (let i = 1; i <= forecastQuarters; i++) {
      const year =
        lastData.year + Math.floor((lastData.quarter + i - 1) / 4);
      const quarter = ((lastData.quarter + i - 1) % 4) + 1;

      const futureRate = lastData["重貼現率_pct"] + i * 0.02;
      const futureMortgage = lastData["房貸餘額_季末"] * (1 + i * 0.01);
      const futureM2 = lastData.M2 * (1 + i * 0.008);
      const futureInventory =
        lastData[inventoryColumn] * (1 + i * 0.015);
      const futureAvgPrice =
        lastData["電梯大廈_六都均價"] * (1 + i * 0.01);

      const futureFeatures = [
        futureRate,
        futureMortgage / 1_000_000,
        futureM2 / 10_000,
        futureInventory / 1_000,
        futureAvgPrice,
        year - 2010,
        Math.sin((quarter * Math.PI) / 2),
        Math.cos((quarter * Math.PI) / 2),
      ];

      const predictedPrice = model.predict(futureFeatures);

      const conf = calculateConfidence(i, model.r2Score);

      futurePoints.push({
        period: `${year}Q${quarter}`,
        year,
        quarter,
        predictedPrice: Math.round(predictedPrice * 100) / 100,
        confidence: conf,
      });
    }

    newPredictions[city] = futurePoints;
  });

  predictions = newPredictions;
  modelMetrics = metrics;
}

// ====== 畫圖 ======

function renderTrendChart() {
  const ctx = document.getElementById("trendChart").getContext("2d");
  const chartData = getChartData();
  if (!chartData.length) return;

  const labels = chartData.map((d) => d.period);
  const actualData = chartData.map((d) =>
    d.actual !== undefined ? d.actual : null
  );
  const predictedData = chartData.map((d) =>
    d.predicted !== undefined ? d.predicted : null
  );
  const upperData = chartData.map((d) =>
    d.upper !== undefined ? d.upper : null
  );
  const lowerData = chartData.map((d) =>
    d.lower !== undefined ? d.lower : null
  );

  if (trendChart) {
    trendChart.destroy();
  }

  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "歷史價格",
          data: actualData,
          borderWidth: 2,
          spanGaps: true,
        },
        {
          label: "Boosting預測",
          data: predictedData,
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: "信心區間上限",
          data: upperData,
          borderWidth: 1,
          borderDash: [4, 4],
          spanGaps: true,
        },
        {
          label: "信心區間下限",
          data: lowerData,
          borderWidth: 1,
          borderDash: [4, 4],
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              if (val == null) return "";
              return `${ctx.dataset.label}: ${val.toFixed(2)} 萬`;
            },
          },
        },
      },
      scales: {
        y: {
          title: {
            display: true,
            text: "單坪價格 (萬元)",
          },
        },
      },
    },
  });
}

function renderFeatureChart() {
  const ctx = document.getElementById("featureChart").getContext("2d");
  const metrics = modelMetrics[selectedCity];
  if (!metrics) return;

  const dataArr = metrics.featureImportance;
  const labels = dataArr.map((d) => d.feature);
  const importance = dataArr.map((d) => d.importance);

  if (featureChart) {
    featureChart.destroy();
  }

  featureChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "使用率 (%)",
          data: importance,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "x",
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) =>
              `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} %`,
          },
        },
      },
      scales: {
        y: {
          title: {
            display: true,
            text: "使用率 (%)",
          },
        },
      },
    },
  });
}

function renderTreeChart() {
  const ctx = document.getElementById("treeChart").getContext("2d");
  const metrics = modelMetrics[selectedCity];
  if (!metrics) return;

  const d = metrics.treeContributions;
  const labels = d.map((x) => `樹 ${x.tree}`);
  const vals = d.map((x) => x.contribution);

  if (treeChart) {
    treeChart.destroy();
  }

  treeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "殘差改善",
          data: vals,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        y: {
          title: {
            display: true,
            text: "殘差改善",
          },
        },
      },
    },
  });
}

// ====== 其他 UI 更新 ======

function renderStats() {
  const latestPrice = getLatestPrice();
  const predictedChange = getPredictedChange();
  const cityMetrics = modelMetrics[selectedCity];

  const latestPriceValue = document.getElementById("latestPriceValue");
  const latestPricePeriod = document.getElementById("latestPricePeriod");
  const futurePriceValue = document.getElementById("futurePriceValue");
  const futurePricePeriod = document.getElementById("futurePricePeriod");
  const changeValue = document.getElementById("changeValue");
  const r2Value = document.getElementById("r2Value");
  const maeValue = document.getElementById("maeValue");
  const maeTreesNote = document.getElementById("maeTreesNote");
  const headerMetaText = document.getElementById("headerMetaText");
  const trendTitle = document.getElementById("trendTitle");

  if (latestPrice != null) {
    latestPriceValue.textContent = `${latestPrice.toFixed(2)} 萬`;
    latestPricePeriod.textContent = "最新一期實際價格";
  } else {
    latestPriceValue.textContent = "--";
  }

  if (predictedChange != null && predictions[selectedCity]) {
    futurePriceValue.textContent =
      predictedChange.futurePrice.toFixed(2) + " 萬";
    futurePricePeriod.textContent =
      predictions[selectedCity][forecastQuarters - 1].period;

    const changeNum = parseFloat(predictedChange.change);
    const sign = changeNum > 0 ? "+" : "";
    changeValue.textContent = `${sign}${predictedChange.change}%`;
    changeValue.classList.toggle("text-green", changeNum > 0);
    changeValue.classList.toggle("text-red", changeNum < 0);
  } else {
    futurePriceValue.textContent = "--";
    futurePricePeriod.textContent = "-";
    changeValue.textContent = "--";
  }

  if (cityMetrics) {
    r2Value.textContent = (cityMetrics.r2Score * 100).toFixed(2) + "%";
    maeValue.textContent = `±${cityMetrics.mae.toFixed(2)} 萬`;
    maeTreesNote.textContent = `${cityMetrics.nTrees} 棵決策樹的集成預測誤差`;
    headerMetaText.textContent = `已訓練 ${cityMetrics.nTrees} 棵決策樹 | 每棵深度: 3 層 | 學習率: 0.1`;
  }

  trendTitle.textContent = `${selectedCity} 歷史與預測趨勢`;
}

function renderPredictionTable() {
  const tbody = document.getElementById("predictionTableBody");
  tbody.innerHTML = "";

  if (!predictions[selectedCity]) return;

  const latestPrice = getLatestPrice();

  predictions[selectedCity].forEach((pred, idx) => {
    const row = document.createElement("tr");

    const prevPrice =
      idx === 0 ? latestPrice : predictions[selectedCity][idx - 1].predictedPrice;
    let change = "--";
    let changeNum = null;
    if (prevPrice) {
      changeNum = ((pred.predictedPrice - prevPrice) / prevPrice) * 100;
      change = (changeNum >= 0 ? "+" : "") + changeNum.toFixed(2) + "%";
    }

    row.innerHTML = `
      <td>${pred.period}</td>
      <td class="text-right"><strong>${pred.predictedPrice.toFixed(
        2
      )}</strong></td>
      <td class="text-right">${pred.confidence.toFixed(1)}%</td>
      <td class="text-right ${
        changeNum != null
          ? changeNum > 0
            ? "text-green"
            : "text-red"
          : ""
      }">${change}</td>
    `;

    tbody.appendChild(row);
  });
}

function hideLoader() {
  const loader = document.getElementById("loader");
  if (loader) loader.style.display = "none";
}

function showLoader() {
  const loader = document.getElementById("loader");
  if (loader) loader.style.display = "flex";
}

function renderAll() {
  renderStats();
  renderTrendChart();
  renderFeatureChart();
  renderTreeChart();
  renderPredictionTable();
}

// ====== 資料載入 ======

function loadAndProcessDataFromString(fileData) {
  Papa.parse(fileData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => {
      console.log("欄位名稱:", results.meta.fields);      // ⬅ 新增
      console.log("第一筆資料:", results.data[0]);        // ⬅ 新增
      const parsedData = results.data.map((row) => ({
        ...row,
        period: `${row.year}Q${row.quarter}`,
      }));

      data = parsedData;
      trainAndPredict(data);
      loading = false;
      hideLoader();
      renderAll();
    },
  });
}

// Electron / preload 提供 window.fs 版本
function loadAndProcessData() {
  showLoader();

  fetch("merged_elevator_quarterly.csv")
    .then((response) => {
      if (!response.ok) {
        throw new Error("CSV 載入失敗，HTTP 狀態碼：" + response.status);
      }
      return response.text();
    })
    .then((fileData) => {
      loadAndProcessDataFromString(fileData); // 這個函數不用改
    })
    .catch((err) => {
      console.error("載入 CSV 發生錯誤：", err);
      hideLoader();
    });
}

// ====== 事件綁定 ======

document.addEventListener("DOMContentLoaded", () => {
  const citySelect = document.getElementById("citySelect");
  const forecastRange = document.getElementById("forecastRange");
  const forecastLabel = document.getElementById("forecastLabel");

  citySelect.addEventListener("change", (e) => {
    selectedCity = e.target.value;
    renderAll();
  });

  forecastRange.addEventListener("input", (e) => {
    forecastQuarters = parseInt(e.target.value, 10);
    forecastLabel.textContent = forecastQuarters;
  });

  forecastRange.addEventListener("change", () => {
    if (data.length) {
      showLoader();
      // 重新訓練（因為預測步數改變）
      trainAndPredict(data);
      hideLoader();
      renderAll();
    }
  });

  // 如果你想改成用 <input type="file"> 上傳，就打開下面註解：
  /*
  const csvInput = document.getElementById('csvFileInput');
  csvInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    showLoader();
    const reader = new FileReader();
    reader.onload = (ev) => {
      loadAndProcessDataFromString(ev.target.result);
    };
    reader.readAsText(file, 'utf-8');
  });
  */

  // 預設：用 window.fs 載入 merged_elevator_quarterly.csv
  loadAndProcessData();
});
