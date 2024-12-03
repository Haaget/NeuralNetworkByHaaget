namespace NeuralNetwork
{
    public class NeuralNetworks
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetworks(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayers();
        }

        public Neuron Predict(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                // Returns the neuron with the largest output value
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        public double Learn(double[] expected, double[,] inputs, int epoch, double errorThreshold = 0.001)
        {
            var totalError = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);
                    totalError += Backpropagation(output, input);
                }

                if (totalError / (i + 1) < errorThreshold)
                {
                    break;
                }
            }

            return totalError / epoch;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var col = matrix.GetLength(1);
            var array = new double[col];
            for (int i = 0; i < col; i++)
            {
                array[i] = matrix[row, i];
            }

            return array;
        }

        private double[,] Scalling(double[,] inputs)
        {
            var res = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int col = 0; col < inputs.GetLength(1); col++)
            {
                var min = inputs[0, col];
                var max = inputs[0, col];

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, col];

                    if (item < min)
                    {
                        min = item;
                    }

                    if (item > max)
                    {
                        max = item;
                    }
                }

                var divider = max - min;
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    res[row, col] = (inputs[row, col] - min) / divider;
                }
            }

            return res;
        }

        private double[,] Normalization(double[,] inputs)
        {
            var res = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int col = 0; col < inputs.GetLength(1); col++)
            {
                // Neuron signal average
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, col];
                }
                var average = sum / inputs.GetLength(0);

                // Standard deviation of a neuron
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, col] - average), 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    res[row, col] = (inputs[row, col] - average) / standartError;
                }
            }

            return res;
        }

        private double Backpropagation(double expected, params double[] inputs)
        {
            var actual = Predict(inputs).Output;
            var difference = actual - expected;

            var weight = expected == 1 ? 3.0 : 1.0;
            difference *= weight;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                var currentLayer = Layers[i];
                var nextLayer = Layers[i + 1];

                for (int j = 0; j < currentLayer.Neurons.Count; j++)
                {
                    var neuron = currentLayer.Neurons[j];

                    double error = 0.0;
                    for (int k = 0; k < nextLayer.Neurons.Count; k++)
                    {
                        var nextNeuron = nextLayer.Neurons[k];
                        error += nextNeuron.Weights[j] * nextNeuron.Delta;
                    }

                    neuron.Learn(error, Topology.LearningRate);
                }
            }

            return difference * difference;
        }

        /// <summary>
        /// Propagates the signals through each layer, starting from the second layer,
        /// using the output values of the previous layer.
        /// </summary>
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        /// <summary>
        /// Assigns the input signals to the neurons in the input layer.
        /// </summary>
        /// <param name="inputSignals">The input values to be fed into the input layer neurons.</param>
        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayers()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int i = 0; i < Topology.HiddenLayerSizes.Count; i++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();

                for (int j = 0; j < Topology.HiddenLayerSizes[i]; j++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();

            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }

            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
    }
}

