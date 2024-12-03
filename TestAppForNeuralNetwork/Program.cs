using NeuralNetwork;

namespace TestAppForNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var topology = new Topology(3, 1, 0.1, 2);

            var neuralNetwork = new NeuralNetworks(topology);

            double[,] inputs = {
                { 0, 0, 0 },
                { 0, 0, 1 },
                { 0, 1, 0 },
                { 1, 0, 0 },
                { 1, 1, 0 },
                { 0, 1, 1 },
                { 1, 0, 1 },
                { 1, 1, 1 },

            };
            double[] expected = { 0, 1, 0, 1, 0, 0, 1, 1 };

            int epoch = 10000;
            var error = neuralNetwork.Learn(expected, inputs, epoch);

            Console.WriteLine($"Error after {epoch} epochs: {error}");

            Console.WriteLine("Predictions:");
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                var inputSignals = new double[] { inputs[i, 0], inputs[i, 1], inputs[i, 2] };
                var prediction = neuralNetwork.Predict(inputSignals);
                Console.WriteLine($"Input: ({inputSignals[0]}, {inputSignals[1]}, {inputSignals[2]}) => Prediction: {prediction.Output}");
            }
            Console.ReadLine();
        }
    }
}