namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType Type { get; }

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            if (neurons == null)
            {
                throw new ArgumentNullException(nameof(neurons), "The input list cannot be empty.");
            }

            if (!Enum.IsDefined(type))
            {
                throw new ArgumentOutOfRangeException(nameof(type), "Invalid neuron type.");
            }

            if (neurons.Count == 0)
            {
                throw new ArgumentException("The neuron list cannot be empty.", nameof(neurons));
            }

            Neurons = neurons;
            Type = type;
        }

        /// <summary>
        /// Gets the output signals of all neurons in the layer.
        /// </summary>
        /// <returns>List of signals.</returns>
        public List<double> GetSignals()
        {
            return Neurons.Select(neuron => neuron.Output).ToList();
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
