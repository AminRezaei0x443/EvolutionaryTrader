import numpy as np


class GeneFactory:
    alphabet = ["S", "T", "F", "R"]
    GENE_SIZE = 12

    @staticmethod
    def decodeAlphabet(code):
        n = len(code) - 1
        b = len(GeneFactory.alphabet)
        r = 0
        for c in code:
            r += GeneFactory.alphabet.index(c) * (b ** n)
            n -= 1
        return r

    @staticmethod
    def encodeAlphabet(val, fieldSize):
        r = np.base_repr(val, base=len(GeneFactory.alphabet))
        for i in range(fieldSize - len(r)):
            r = "0" + r
        code = ""
        for d in r:
            code += GeneFactory.alphabet[int(d)]
        return code

    @staticmethod
    def config2Gene(config):
        gene = ""
        gene += GeneFactory.encodeAlphabet(config["pivots"], 2)
        gene += GeneFactory.encodeAlphabet(config["timeFrame"], 1)
        gene += GeneFactory.encodeAlphabet(config["diffMajor"], 2)
        gene += GeneFactory.encodeAlphabet(config["diffMinor"], 4)
        gene += GeneFactory.encodeAlphabet(config["batch"], 3)
        return gene

    @staticmethod
    def validateGene(gene):
        if len(gene) != GeneFactory.GENE_SIZE:
            return False
        for g in gene:
            if g not in GeneFactory.alphabet:
                return False
        return True

    @staticmethod
    def gene2Config(gene):
        if not GeneFactory.validateGene(gene):
            return None
        return {
            "pivots": GeneFactory.decodeAlphabet(gene[0:2]),
            "timeFrame": GeneFactory.decodeAlphabet(gene[2:3]),
            "diffMajor": GeneFactory.decodeAlphabet(gene[3:5]),
            "diffMinor": GeneFactory.decodeAlphabet(gene[5:9]),
            "batch": GeneFactory.decodeAlphabet(gene[9:12]),
        }

    @staticmethod
    def newGene(parentI, parentII, mutationRate=0.075):
        coPoint = np.random.randint(GeneFactory.GENE_SIZE, size=1)[0]
        mix = parentI[0:coPoint] + parentII[coPoint:]
        mutationMask = np.random.rand(GeneFactory.GENE_SIZE) < mutationRate
        child = ""
        for i, m in enumerate(mutationMask):
            if m:
                child += GeneFactory.alphabet[(len(GeneFactory.alphabet) - 1) - GeneFactory.alphabet.index(mix[i])]
            else:
                child += mix[i]
        return child

    @staticmethod
    def randGene(mutationRate=0.075):
        mix = ""
        r = np.random.randint(len(GeneFactory.alphabet), size=GeneFactory.GENE_SIZE)
        for d in r:
            mix += GeneFactory.alphabet[int(d)]
        mutationMask = np.random.rand(GeneFactory.GENE_SIZE) < mutationRate
        child = ""
        for i, m in enumerate(mutationMask):
            if m:
                child += GeneFactory.alphabet[(len(GeneFactory.alphabet) - 1) - GeneFactory.alphabet.index(mix[i])]
            else:
                child += mix[i]
        return child
