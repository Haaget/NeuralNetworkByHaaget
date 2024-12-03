namespace NeuralNetwork
{
    public class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public double LearningRate { get; }
        public List<int> HiddenLayerSizes { get; }

        public Topology(int inputCount, int outputCount, double learningRate, params int[] hiddenLayers)
        {
            if (inputCount <= 0)
                throw new ArgumentException("The number of input neurons must be greater than zero.", nameof(inputCount));

            if (outputCount <= 0)
                throw new ArgumentException("The number of output neurons must be greater than zero.", nameof(outputCount));

            if (hiddenLayers == null || hiddenLayers.Length == 0)
                throw new ArgumentException("The network must have at least one hidden layer.", nameof(hiddenLayers));

            if (hiddenLayers.Any(layer => layer <= 0))
                throw new ArgumentException("All neuron values must be greater than zero.", nameof(hiddenLayers));

            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayerSizes = new List<int>(hiddenLayers);
        }

        public int GetTotalLayerCount()
        {
            return 1 + HiddenLayerSizes.Count + 1;
        }

        public List<int> GetLayerSizes()
        {
            var layers = new List<int> { InputCount };
            layers.AddRange(HiddenLayerSizes);
            layers.Add(OutputCount);
            return layers;
        }

        public override string ToString()
        {
            var layerSizes = string.Join(" -> ", GetLayerSizes());
            return $"Topology: {layerSizes}";
        }
    }
}

