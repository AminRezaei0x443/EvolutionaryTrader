from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CryptoAgent:
    def __init__(self, pivots=3, pivotDiff=0.005, batchNum=1):
        self.pivotsN = pivots
        self.pivotDiff = pivotDiff
        self.prices = np.array([])
        self.events = []
        self.position = None
        self.positionHistory = []
        self.lastPrice = 0
        self.stopLoss = 0
        self.logs = []
        self.batchNum = batchNum
        self.batchC = 0

    def feed(self, data):
        self.prices = np.concatenate((self.prices, data))
        self.margin = len(self.prices)
        self.kmeans = KMeans(n_clusters=self.pivotsN, random_state=0)
        self.kmeans.fit(self.prices.reshape(-1, 1))
        self.pivots = self.kmeans.cluster_centers_.reshape(-1)
        self.pivotHighs = self.pivots * (1 + self.pivotDiff)
        self.pivotLows = self.pivots * (1 - self.pivotDiff)

    def findLastEvent(self, pivot):
        for e in self.events[::-1]:
            if e["pivot"] == pivot:
                return e
        return None

    def log(self, msg):
        self.logs.append(msg)

    def beforePrice(self):
        return self.prices[-2]

    def observe(self, price):
        self.prices = np.append(self.prices, price)
        if self.batchC == self.batchNum:
            self.kmeans.fit(self.prices.reshape(-1, 1))
            self.pivots = self.kmeans.cluster_centers_.reshape(-1)
            self.pivotHighs = self.pivots * (1 + self.pivotDiff)
            self.pivotLows = self.pivots * (1 - self.pivotDiff)
            self.batchC = 0
        else:
            self.batchC += 1
        d = price
        for pivot in self.pivots:
            if d > pivot > self.beforePrice():
                e = self.findLastEvent(pivot)
                if e is None or e["kind"] != "UP":
                    self.events.append({
                        "pivot": pivot,
                        "kind": "UP"
                    })
                    self.log("saw %f while going up" % pivot)
            elif d < pivot < self.beforePrice():
                e = self.findLastEvent(pivot)
                if e is None or e["kind"] != "DOWN":
                    self.events.append({
                        "pivot": pivot,
                        "kind": "DOWN"
                    })
                    self.log("saw %f while going down" % pivot)
        for i, high in enumerate(self.pivotHighs):
            if d > high > self.beforePrice():
                if self.position is None:
                    self.position = {
                        "type": "BUY",
                        "price": high,
                        "index": len(self.prices) - 1
                    }
                    self.stopLoss = self.pivotLows[i]
                    self.log("already in position | breaking up %f " % high)
                    return self.position
                else:
                    self.stopLoss = self.pivotLows[i]
                self.log("breaking up %f " % high)

        for i, low in enumerate(self.pivotLows):
            if d > low > self.beforePrice():
                if self.position is not None:
                    pUp = self.pivots[i]
                    pDown = self.position["price"]
                    self.stopLoss = pDown + (pUp - pDown) * 0.5

        if d < self.stopLoss < self.beforePrice():
            if self.position is not None:
                self.positionHistory.append(self.position)
                act = {
                    "type": "SELL",
                    "price": self.stopLoss,
                    "index": len(self.prices) - 1
                }
                self.positionHistory.append(act)
                self.positionHistory.append({
                    "type": "REPORT",
                    "profit": (self.stopLoss - self.position["price"]) / self.position["price"]
                })
                self.log("selling position | breaking down %f" % self.stopLoss)
                self.position = None
                return act
            self.log("breaking down %f" % self.stopLoss)

    def plotPivots(self, savePath=None):
        sns.set(style="darkgrid", context="talk")
        plt.style.use("dark_background")
        plt.rcParams.update({"grid.linewidth": 0.5, "grid.alpha": 0.5})

        sns.lineplot(data=self.prices, color="#FFF689", lw=1)
        plt.axvline(self.margin, color="#9EE493", linestyle=":")
        for pivot in self.pivots:
            plt.axhline(pivot, color="#ff0099", linestyle="--", lw=1)
            plt.axhline(pivot * (1 + self.pivotDiff), color="#3DB1F5", linestyle="--", lw=0.6)
            plt.axhline(pivot * (1 - self.pivotDiff), color="#3DB1F5", linestyle="--", lw=0.6)
        if savePath is not None:
            plt.savefig(savePath)
        plt.clf()

    def plotOrders(self, savePath=None):
        sns.lineplot(data=self.prices, color="#FFF689", lw=1)
        for position in self.positionHistory:
            if position["type"] == "BUY":
                plt.axvline(position["index"], color="#9EE493", lw=1)
            if position["type"] == "SELL":
                plt.axvline(position["index"], color="#3DB1F5", lw=1)
        if savePath is not None:
            plt.savefig(savePath)
        plt.clf()

    def report(self, printResult=False):
        overall = 1
        failed = 0
        success = 0
        for position in self.positionHistory:
            if position["type"] == "REPORT":
                overall *= (1 + position["profit"])
                if position["profit"] > 0:
                    success += 1
                else:
                    failed += 1
        if printResult:
            print("Overall profit: %f" % (overall - 1))
            print("Successful positions: %d | Failed Positions: %d" % (success, failed))
        return {
            "profit": overall - 1,
            "success": success,
            "failure": failed
        }

    def state(self) -> dict:
        return {
            "pivotsN": self.pivotsN,
            "pivotDiff": self.pivotDiff,
            "prices": self.prices,
            "events": self.events,
            "position": self.position,
            "positionHistory": self.positionHistory,
            "lastPrice": self.lastPrice,
            "logs": self.logs,
            "margin": self.margin,
            "pivots": self.pivots,
            "pivotHighs": self.pivotHighs,
            "pivotLows": self.pivotLows
        }

    def loadState(self, stateDict):
        self.pivotsN = stateDict["pivotsN"]
        self.pivotDiff = stateDict["pivotDiff"]
        self.prices = stateDict["prices"]
        self.events = stateDict["events"]
        self.position = stateDict["position"]
        self.positionHistory = stateDict["positionHistory"]
        self.lastPrice = stateDict["lastPrice"]
        self.logs = stateDict["logs"]
        self.margin = stateDict["margin"]
        self.pivots = stateDict["pivots"]
        self.pivotHighs = stateDict["pivotHighs"]
        self.pivotLows = stateDict["pivotLows"]
