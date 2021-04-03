from data_store import DataStore
from agent import CryptoAgent
from genetics import GeneFactory
import numpy as np
import numpy.random as nrand


if __name__ == "__main__":
    # Config
    symbol = "dotusdt"
    dataCount = 1000
    c = int(0.2 * dataCount)
    frames = ["1m", "15m", "30m", "1h"]
    minorStep = 1 / 257

    # First Generation
    population = []
    initGenes = [
        GeneFactory.config2Gene({'pivots': 3, 'timeFrame': 0, 'diffMajor': 0, 'diffMinor': 1, 'batch': 15}),
        GeneFactory.config2Gene({'pivots': 5, 'timeFrame': 1, 'diffMajor': 0, 'diffMinor': 5, 'batch': 1}),
    ]
    gId = 0
    for g in initGenes:
        population.append((gId, g, 1, None))
        gId += 1

    # History ("Id", "Gene", "Generation", "Parents")
    populationHistory = []
    scoreHistory = {}

    generation = 1
    targetGeneration = 2
    while generation != targetGeneration + 1:
        print("processing generation: ", generation)
        weights = []
        for geneId, gene, _, _ in population:
            # Read Config And Create Agent
            config = GeneFactory.gene2Config(gene)
            diff = (config["diffMinor"] * minorStep)
            agent = CryptoAgent(config["pivots"] + 1, diff, config["batch"] + 1)
            # Read Data
            data = DataStore.loadKlineClosesHistory(symbol, frames[config["timeFrame"]], dataCount)
            # Feed and Take Action using data train | test split
            agent.feed(data[:c])
            for d in data[c:]:
                act = agent.observe(d)
            # Record And Display Results
            weights.append(agent.report(printResult=False)["profit"])
            print("processing id:", geneId, " gene: ", gene, " | score: ", weights[-1])
            scoreHistory[geneId] = weights[-1]

        # Handle Negative Scores and Calculate Probability
        mS = np.min(weights)
        if mS < 0:
            weights += 2 * -1 * mS
        probabilities = np.array(weights) / np.sum(weights)

        # Generate New Generation
        newPopulation = []
        newGenes = []
        for _ in range(len(population)):
            i1, i2 = tuple(nrand.choice(range(len(population)), size=2, replace=False, p=probabilities))
            gx1, gx2 = population[i1], population[i2]
            p1, p2 = gx1[1], gx2[1]

            # Reproduce Genes until no duplicate exists
            g1 = GeneFactory.newGene(p1, p2)
            while g1 in newGenes:
                g1 = GeneFactory.newGene(p1, p2)
            newPopulation.append((gId, g1, generation + 1, (gx1[0], gx2[0])))
            newGenes.append(g1)
            gId += 1

            g2 = GeneFactory.newGene(p2, p1)
            while g2 in newGenes:
                g2 = GeneFactory.newGene(p2, p1)
            newPopulation.append((gId, g2, generation + 1, (gx2[0], gx1[0])))
            newGenes.append(g2)
            gId += 1
        populationHistory += population
        population = newPopulation
        generation += 1

    bestGeneId = max(scoreHistory.items(), key=lambda x: x[1])[0]
    for geneO in populationHistory:
        geneId, gene, generation, _ = geneO
        if geneId == bestGeneId:
            print("Best Gene Found!")
            print("Gene Id: ", geneId)
            print("Genetic Code: ", gene)
            print("Generation: ", generation)
