import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from src.dataset import Dataset

toy_example = Dataset()
print(toy_example.n)
P = toy_example.generate_data()
print(P.shape)

if __name__ == "__main__":
    conf = SparkConf().setAppName("word count").setMaster("local[2]")
    sc = SparkContext(conf=conf)

    lines = sc.textFile("word_count.txt")
    words = lines.flatMap(lambda line: line.split(" "))
    wordCounts = words.countByValue()

    for word, count in wordCounts.items():
        print("{} : {}".format(word, count))
