namespace NeuralNetwork
{
    public class Neuron
    {
        private static Random rnd = new Random();

        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        // private ActivationFunctionType _activationFunction;

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            if (inputCount <= 0)
            {
                throw new ArgumentException("The value must be a positive number.", nameof(inputCount));
            }

            if (!Enum.IsDefined(typeof(NeuronType), type))
            {
                throw new ArgumentOutOfRangeException(nameof(type), "Invalid neuron type.");
            }

            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {
            Weights.Clear();
            Inputs.Clear();

            for (int i = 0; i < inputCount; i++)
            {
                double weight = NeuronType == NeuronType.Input ? 1 : (rnd.NextDouble() * 2.0) - 1.0;
                Weights.Add(weight);
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs), "The input list cannot be empty.");
            }

            if (Weights.Count != inputs.Count)
            {
                throw new ArgumentException("The number of input data must match the number of weights.");
            }

            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            Output = NeuronType == NeuronType.Input ? sum : Sigmoid(sum);

            return Output;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidDx(double x) => x * (1.0 - x);

        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var newWeight = Weights[i] - Inputs[i] * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString() => Output.ToString();
    }
}
