/**
 * This module creates types and functions for the purpose of defining an A-B-1 neural network. The module includes defining
 * structures that encase network parameters (learning rate, number of hidden nodes, number of input nodes, number of output nodes
 * which is coded to 1, training mode, weight initialization, random weight generation lower and upper bounds, error threshold to
 * stop training, and max iterations for training). Furthermore, the NetworkArrays struct defines the arrays and weights following
 * the allocation of these structures.
 * The main function sets the network parameters using 'setNetworkParameters()', echos the parameters through console using
 * 'echoNetworkParameters()', allocates the memory needed for the network arrays using 'allocateNetworkMemory()', populates
 * the network arrays by initializing the weights using 'populateNetworkMemory()', trains the network using 'train()', and
 * runs the network irrespective of the train mode using 'run()' for every row of the truth table.
 * This module includes functions that train and run the network for a given truth table, as well as "setup" functions that
 * create, allocate, and populate the A-B-1 neural network. The training algorithm utilizes gradient descent to minimize the
 * error function.
 * @author Daniel Gergov
 * @creation 1/27/24
 */

package main

import
(
   "fmt"  // import the io interface
   "math" // import the math library for max integer
   "math/rand"
   "time" // import time to measure the execution time
) // import 

type NetworkParameters struct
{
   learningRate float64
   
   numHiddenNodes int
   numInputNodes  int
   numOutputNodes int
   numTestCases   int
   
   trainMode  bool
   weightInit int // 1 is randomize, 2 is zero
   
   weightLowerBound float64
   weightUpperBound float64
   
   errorThreshold float64
   maxIterations  int
} // type NetworkParameters struct

type NetworkArrays struct
{
   a                        *[]float64
   h                        *[]float64
   inputHiddenWeights       *[][]float64
   hiddenOutputWeights      *[]float64
   outputNode               float64
   thetas                   *[]float64
   omegas                   *[]float64
   psis                     *[]float64
   dEdW                     *[][]float64
   inputHiddenDeltaWeights  *[][]float64
   hiddenOutputDeltaWeights *[]float64
} // type NetworkArrays struct

var parameters NetworkParameters
var arrays NetworkArrays
var truthTable [][]float64
var expectedOutputs []float64

/**
 * The main function initiates network parameter configuration, memory allocation for network operations, and executes
 * network training and testing. It follows these steps:
 * 1. Sets and displays network parameters.
 * 2. Allocates and populates network memory for arrays, truth table, and expected outputs.
 * 3. Trains the network with provided truth table and expected outputs if the trainMode network configuration is 'true'
 * 4. Iterates over the truth table, comparing and outputting expected outputs against predictions for each input.
 *
 * Limitations:
 * - Assumes correct implementation of and depends on external functions (setNetworkParameters, echoNetworkParameters,
 *    allocateNetworkMemory, populateNetworkMemory, train, run) without error handling.
 * - Requires that truthTable and expectedOutputs are pre-declared and their sizes match, ensuring each input
 *    correlates with its expected output.
 */
func main()
{
   setNetworkParameters()
   echoNetworkParameters()
   
   arrays, truthTable, expectedOutputs = allocateNetworkMemory()
   populateNetworkMemory()
   
   if (parameters.trainMode)
   {
      train(truthTable, expectedOutputs)
   }
   
   var input []float64
   var index int
   for index, input = range truthTable
   {
      fmt.Printf("Input: %v, Expected: %f, Predicted: %f\n", input, expectedOutputs[index], run(input))
   }
} // func main()

/**
 * The setNetworkParameters function initializes the network's configuration by defining its parameters within a
 * NetworkParameters structure. This setup includes learning rate, number of nodes in each layer
 * (input, hidden, output), number of test cases, training mode status, weight initialization method, bounds for weight values,
 * error threshold, and the maximum number of iterations.
 *
 * Parameters Defined:
 * - learningRate: Learning rate (lambda) for the gradient descent rate.
 * - numInputNodes, numHiddenNodes, numOutputNodes: Specify the architecture of the network in terms of neuron counts.
 * - numTestCases: The number of test cases to be used in training/validation.
 * - trainMode: Boolean indicating if the network is in training mode.
 * - weightInit: Method or value for weight initialization; 1 being random, 2 being zeroes.
 * - weightLowerBound, weightUpperBound: Define the range of values for initializing the network weights to random values.
 * - errorThreshold: The error level at which training is considered sufficiently complete.
 * - maxIterations: Limits the number of training iterations.
 *
 * Limitations:
 * - The function statically sets network parameters without accepting external input.
 */
func setNetworkParameters()
{
   parameters = NetworkParameters{
      learningRate:     0.3,
      numInputNodes:    2,
      numHiddenNodes:   1,
      numOutputNodes:   1,
      numTestCases:     4,
      trainMode:        true,
      weightInit:       1,
      weightLowerBound: -1.5,
      weightUpperBound: 1.5,
      errorThreshold:   2e-4,
      maxIterations:    100000,
   }
} // func setNetworkParameters()

/**
 * The echoNetworkParameters function prints the current network configuration parameters to the console. The function clearly
 * displays the network's configurations including learning rate (λ), error threshold, maximum number of iterations,
 * network architecture (input-hidden-output node counts), number of test cases, training mode status, weight initialization
 * method, and the range for random weight initialization.
 *
 * Displayed Parameters:
 * - Learning rate (λ), error threshold, and maximum iterations for training control.
 * - Network architecture detailed by the count of input, hidden, and output nodes.
 * - The total number of test cases used for training or validation.
 * - Training mode indicator (true for training mode).
 * - Weight initialization method, where "1" denotes random initialization and "2" denotes initialization to zero.
 * - Range for random weight initialization, specifying lower and upper bounds.
 *
 * Limitations:
 * - This function depends on the global `parameters` structure being set by setNetworkParameters() and not by passed arguments.
 */
func echoNetworkParameters()
{
   fmt.Println("Network Parameters:")
   fmt.Println("-------------------")
   fmt.Printf("λ: %v, Error Threshold: %v, Max Iterations: %d\n",
      parameters.learningRate, parameters.errorThreshold, parameters.maxIterations)
   fmt.Printf("Network: %d-%d-%d, NumberTestCases: %d\n",
      parameters.numInputNodes, parameters.numHiddenNodes, parameters.numOutputNodes, parameters.numTestCases)
   fmt.Printf("Train Mode: %t\n", parameters.trainMode)
   fmt.Printf("Weight Init: %d -- 1 = random, 2 = zero\n", parameters.weightInit)
   fmt.Printf("Random Range [%v, %v]\n\n", parameters.weightLowerBound, parameters.weightUpperBound)
} // func echoNetworkParameters()

/**
 * The allocateNetworkMemory function is responsible for initializing and allocating memory for various arrays and matrices used
 * by the network, including those for input to hidden layer weights, hidden to output layer weights, and other structures
 * for training such as gradients, delta weights, omegas, thetas, and psis. The function also allocates the truth table for
 * inputs and expected outputs.

 * Returns:
 * - A NetworkArrays structure containing references to all allocated arrays and matrices used by the network.
 * - A truth table for network inputs as a slice of float64 slices.
 * - An output truth table as a slice of float64.

 * If the trainMode parameter is true, structures used exclusively for training (thetas, omegas, psis, dEdW, deltaWeights)
 * are allocated. This condition helps optimize memory usage by only allocating necessary arrays.

 * Limitations:
 * - Assumes that the global `parameters` structure is correctly initialized before this function is called.
 */
func allocateNetworkMemory() (NetworkArrays, [][]float64, []float64)
{
   var j, input int
   
   var a []float64 = make([]float64, parameters.numInputNodes)
   var inputHiddenWeights [][]float64 = make([][]float64, parameters.numInputNodes)
   for j = range inputHiddenWeights
   {
      inputHiddenWeights[j] = make([]float64, parameters.numHiddenNodes)
   }

   var h []float64 = make([]float64, parameters.numHiddenNodes)
   var hiddenOutputWeights []float64 = make([]float64, parameters.numHiddenNodes)
   var outputNode float64 = 0.0
   
   var thetas, omegas, psis, hiddenOutputDeltaWeights []float64
   var dEdW, inputHiddenDeltaWeights [][]float64
   
   /* prevent the allocation of thetas, omegas, psis, dEdW, and deltaWeights if not training */
   if (parameters.trainMode)
   {
      thetas = make([]float64, parameters.numHiddenNodes)
      omegas = make([]float64, parameters.numHiddenNodes)
      psis = make([]float64, parameters.numHiddenNodes)
      
      dEdW = make([][]float64, parameters.numInputNodes)
      for j = range dEdW
      {
         dEdW[j] = make([]float64, parameters.numHiddenNodes)
      }
      
      hiddenOutputDeltaWeights = make([]float64, parameters.numHiddenNodes)
      
      inputHiddenDeltaWeights = make([][]float64, parameters.numInputNodes)
      for j = range inputHiddenDeltaWeights
      {
         inputHiddenDeltaWeights[j] = make([]float64, parameters.numHiddenNodes)
      }
   } // if (parameters.trainMode)
   
   var inputTruthTable [][]float64 = make([][]float64, parameters.numTestCases)
   for input = range inputTruthTable
   {
      inputTruthTable[input] = make([]float64, parameters.numInputNodes)
   }
   
   var outputTruthTable []float64 = make([]float64, parameters.numTestCases)
   
   return NetworkArrays
   {
      a:                              &a,
      h:                              &h,
      inputHiddenWeights:             &inputHiddenWeights,
      hiddenOutputWeights:            &hiddenOutputWeights,
      outputNode:                     outputNode,
      thetas:                         &thetas,
      omegas:                         &omegas,
      psis:                           &psis,
      dEdW:                           &dEdW,
      inputHiddenDeltaWeights:        &inputHiddenDeltaWeights,
      hiddenOutputDeltaWeights:       &hiddenOutputDeltaWeights,
   }, inputTruthTable, outputTruthTable
} // func allocateNetworkMemory() NetworkArrays

/**
 * The populateNetworkMemory function initializes the network's weight matrices and sets up the truth table and expected outputs
 * for training or evaluation. It follows these steps:
 *
 * 1. If the weight initialization mode is set to random (weightInit == 1), it initializes the input-hidden and hidden-output
 * weight matrices with random values within the bounds (weightLowerBound to weightUpperBound).
 * 2. Resets the output node value to 0.0 as a default starting condition.
 * 3. Populates the truth table with predefined inputs.
 * 4. Sets the expected outputs corresponding to the truth table inputs to a binary operation either XOR, OR, or AND.
 *
 * Limitations:
 * - Assumes `arrays`, `truthTable`, and `expectedOutputs` are globally accessible and correctly linked to the network's structure.
 */
func populateNetworkMemory()
{
   var k int
   var j int
   
   if (parameters.weightInit == 1)
   {
      rand.Seed(time.Now().UnixNano())
      for k = range *arrays.inputHiddenWeights
      {
         for j = range (*arrays.inputHiddenWeights)[k]
         {
            (*arrays.inputHiddenWeights)[k][j] = parameters.weightLowerBound +
                                                 rand.Float64()*(parameters.weightUpperBound - parameters.weightLowerBound)
         }
      }
      for k = range *arrays.hiddenOutputWeights
      {
         (*arrays.hiddenOutputWeights)[k] = parameters.weightLowerBound +
                                            rand.Float64()*(parameters.weightUpperBound - parameters.weightLowerBound)
      }
   } // if (parameters.weightInit == 1)
   
   arrays.outputNode = 0.0
   
   truthTable[0][0] = 0.0
   truthTable[0][1] = 0.0
   truthTable[1][0] = 0.0
   truthTable[1][1] = 1.0
   truthTable[2][0] = 1.0
   truthTable[2][1] = 0.0
   truthTable[3][0] = 1.0
   truthTable[3][1] = 1.0
   
   expectedOutputs[0] = 0.0
   expectedOutputs[1] = 1.0
   expectedOutputs[2] = 1.0
   expectedOutputs[3] = 0.0
} // func populateNetworkMemory()

/**
 * The sigmoid function calculates the sigmoid activation of a given input value `x`. The sigmoid activation function
 * introduces non-linearity into a model.
 *
 * The sigmoid of `x` follows the formula:
 * sigmoid(x) = 1 / (1 + e^(-x))
 *
 * Parameters:
 * - x: The input value for which to apply sigmoid formula.
 *
 * Returns:
 * - The sigmoid activation of `x`, a float64 value in the range (0, 1).
 */
func sigmoid(x float64) float64
{
   return 1.0 / (1.0 + math.Exp(-x))
}

/**
 * The sigmoidDerivative function computes the derivative of the sigmoid activation function for a given input value `x`.
 * The derivative is used the calculation for the delta weights through computing the psi variables. The function first
 * calculates the sigmoid of `x`, then applies the derivative formula of the sigmoid function.
 *
 * The derivative of the sigmoid function S(x) is given by:
 * S'(x) = S(x) * (1 - S(x))
 * where S(x) is the sigmoid of `x`.
 *
 * Parameters:
 * - x: The input value for which to apply the sigmoid derivative formula.
 *
 * Returns:
 * - The derivative of the sigmoid function of `x`, a float64 value in the range (0, 0.25].
 */
func sigmoidDerivative(x float64) float64
{
   x = sigmoid(x)
   return x * (1.0 - x)
}

/**
 * The activationFunction function acts as a wrapper to the current activation function being used. Such a function increases
 * the usability of the code as one can change the activation function at once place in the source code.
 *
 * Parameters:
 * - x: The input value for which to apply the activation function to.
 *
 * Returns:
 * - The output of the activation function
 */
func activationFunction(x float64) float64
{
   return sigmoid(x)
}

/**
 * The activationPrime function acts as a wrapper to the current activation function's derivative being used. Such a function
 * increases the usability of the code as one can change the activation function derivative at once place in the source code.
 *
 * Parameters:
 * - x: The input value for which to apply the activation function's derivative to.
 *
 * Returns:
 * - The output of the activation function's derivative
 */
func activationPrime(x float64) float64
{
   return sigmoidDerivative(x)
}

/**
 * The train function runs the neural network's training process using a given set of inputs and their expected outputs.
 * It uses a loop to adjust the network's weights based on the error between predicted and expected outputs, using
 * forward and backward propagation for each input case until the average error across the test cases falls below an error
 * threshold or the maximum number of iterations is reached.
 *
 * Process Overview:
 * 1. Initializes training parameters and stopping condition variables.
 * 2. Enters a training loop that loops over each input:
 *    - Performs forward propagation to compute the network's output.
 *    - Executes backward propagation to calculate gradients and to find the delta weights for every layer.
 *    - Performs another forward propagation to update the error and hidden activations
 * 3. Updates training error and checks the stopping criteria (error threshold or max iterations).
 * 4. Concludes training when stopping criteria are satisfied.
 *
 * Parameters:
 * - `inputs`: Array of input vectors for the network.
 * - `expectedOutputs`: Corresponding expected outputs for each input vector.
 *
 * Limitations:
 * - Assumes that there exists global access to network arrays and parameters for weight updates.
 * - Depends on the functions (`runTrain`, `sigmoidDerivative`) to train the network.
 */
func train(inputs [][]float64, expectedOutputs []float64)
{
   var trainStart time.Time = time.Now()
   var done bool
   
   var epochError float64 = math.MaxFloat64
   var epoch int = 0
   var input, j, k int
   var theta0, omega0, psi0 float64
   
   var a []float64 = *arrays.a
   var h []float64 = *arrays.h
   var thetas []float64 = *arrays.thetas
   var omegas []float64 = *arrays.omegas
   var psis []float64 = *arrays.psis
   var inputHiddenWeights [][]float64 = *arrays.inputHiddenWeights
   var hiddenOutputWeights []float64 = *arrays.hiddenOutputWeights
   var dEdW [][]float64 = *arrays.dEdW
   var inputHiddenDeltaWeights [][]float64 = *arrays.inputHiddenDeltaWeights
   var hiddenOutputDeltaWeights []float64 = *arrays.hiddenOutputDeltaWeights
   
   for (!done)
   {
      epochError = 0.0
      for input = range inputs
      {
         /* forward propagation */
         runTrain(&a, &h, &thetas, &inputHiddenWeights, &hiddenOutputWeights, &inputs[input], &theta0)
         
         /* back propagation for output layer */
         omega0 = expectedOutputs[input] - arrays.outputNode
         epochError += 0.5 * omega0 * omega0
         psi0 = omega0 * activationPrime(theta0)
         
         for j = range hiddenOutputWeights
         {
            dEdW[0][j] = -h[j] * psi0
            hiddenOutputDeltaWeights[j] = -parameters.learningRate * dEdW[0][j]
         }
         
         /* back propagation for hidden layer */
         for k = range inputHiddenWeights
         {
            for j = range inputHiddenWeights[k]
            {
               omegas[j] = psi0 * hiddenOutputWeights[j]
               psis[j] = omegas[j] * activationPrime(thetas[j])
               dEdW[k][j] = -a[k] * psis[j]
               inputHiddenDeltaWeights[k][j] = -parameters.learningRate * dEdW[k][j]
            } // for j = range inputHiddenWeights[k]
         } // for k = range inputHiddenWeights
         
         /* update the weights of the network */
         for j = range hiddenOutputWeights
         {
            hiddenOutputWeights[j] += hiddenOutputDeltaWeights[j]
         }
         
         for k = range inputHiddenWeights
         {
            for j = range inputHiddenWeights[k]
            {
               inputHiddenWeights[k][j] += inputHiddenDeltaWeights[k][j]
            }
         }
         
         /* forward propagation again for update the */
         runTrain(&a, &h, &thetas, &inputHiddenWeights, &hiddenOutputWeights, &inputs[input], &theta0)
         
         omega0 = expectedOutputs[input] - arrays.outputNode
         epochError += 0.5 * omega0 * omega0
      } // for input = range inputs
      epochError /= float64(parameters.numTestCases)
      epoch++
      done = epochError < parameters.errorThreshold || epoch > parameters.maxIterations
      if (epoch % 100000 == 0)
      {
         fmt.Printf("Finished epoch %d with error %f.7\n", epoch, epochError)
      }
   } // for (!done)
   
   var executionTime float64 = float64(time.Since(trainStart) / time.Millisecond)
   
   fmt.Printf("Network training stopped after %s and %v iterations: ", formatTime(executionTime), epoch)
   if (epoch >= parameters.maxIterations)
   {
      fmt.Printf("Exceeded max iterations of %d\n", parameters.maxIterations)
   }
   
   if (epochError <= parameters.errorThreshold)
   {
      fmt.Printf("Error became less than error threshold %.7f\n", epochError)
   }
} // func train(inputs [][]float64, expectedOutputs []float64)

/**
 * The run function performs forward propagation through the network for a given input array `a`, computing the network's output.
 * The function uses the input array to compute dot products between the input neurons and the weights, eventually computing the
 * value of the output node (the network's output).
 *
 * Process Overview:
 * 1. Computes weighted sums at the hidden nodes, applying the sigmoid activation function to each sum/theta.
 * 2. Collects the hidden activations, again applying weights and the sigmoid function to produce the final output.
 * 3. Returns the network's prediction for the given input.
 *
 * Parameters:
 * - `a`: Input array to use to make a prediction.
 *
 * Limitations:
 * - Assumes that the network's weights (`inputHiddenWeights` and `hiddenOutputWeights`) have been properly initialized.
 * - Assumes that the input array `a` matches the size expected by the network.
 */
func run(a []float64) float64
{
   var j, k int
   
   var h []float64 = *arrays.h
   var inputHiddenWeights [][]float64 = *arrays.inputHiddenWeights
   var hiddenOutputWeights []float64 = *arrays.hiddenOutputWeights
   
   // forward propagation
   for j = 0; j < parameters.numHiddenNodes; j++
   {
      var sum float64 = 0.0
      for k = range a
      {
         sum += a[k] * inputHiddenWeights[k][j]
      }
      h[j] = activationFunction(sum)
   } // for j = 0; j < parameters.numHiddenNodes; j++
   
   var sum float64 = 0.0
   for j = range *arrays.h
   {
      sum += h[j] * hiddenOutputWeights[j]
   }
   arrays.outputNode = activationFunction(sum)
   
   return arrays.outputNode
} // func run(a []float64) float64

/**
 * The runTrain function is designed for the forward propagation part of the neural network training process.
 * It updates the network's hidden layer outputs and the final output node value based on a given input array.
 *
 * Process:
 *    1. Copies the input vector into the network's input layer.
 *    2. For each hidden neuron, computes the weighted sum of its inputs and applies the sigmoid function to
 *       obtain the neuron's output.
 *    3. Calculates the weighted sum of the hidden layer outputs and applies the sigmoid function to determine the network output.
 *
 * Parameters:
 * - `a`: Reference to the input layer vector.
 * - `h`: Reference to the hidden layer output vector.
 * - `thetas`: Reference to the vector storing the weighted sums of the hidden layer inputs.
 * - `inputHiddenWeights`: Reference to the matrix of weights from the input layer to the hidden layer.
 * - `hiddenOutputWeights`: Reference to the vector of weights from the hidden layer to the output neuron.
 * - `input`: The input vector for the current training example.
 * - `theta0`: Reference to the variable storing the weighted sum of the hidden layer outputs before applying sigmoid.
 *
 * Limitations and Conditions:
 * - Assumes that the network's weights (`inputHiddenWeights` and `hiddenOutputWeights`) have been properly initialized.
 */
func runTrain(a *[]float64, h *[]float64, thetas *[]float64, inputHiddenWeights *[][]float64,
              hiddenOutputWeights *[]float64, input *[]float64, theta0 *float64)
{
   var j, k int
   
   for j = range *a
   {
      (*a)[j] = (*input)[j]
   }
   
   for j = 0; j < parameters.numHiddenNodes; j++
   {
      (*thetas)[j] = 0.0
      for k = range *a
      {
         (*thetas)[j] += (*a)[k] * (*inputHiddenWeights)[k][j]
      }
      (*h)[j] = activationFunction((*thetas)[j])
   }
   *theta0 = 0.0
   
   for j = range *h
   {
      *theta0 += (*h)[j] * (*hiddenOutputWeights)[j]
   }
   arrays.outputNode = activationFunction(*theta0)
} // func runTrain(a *[]float64...

/**
 * The formatTime function converts a duration in milliseconds into a readable string format, choosing the most appropriate
 * unit (milliseconds, seconds, minutes, hours, days, or weeks) based on the duration's length. The function is used for
 * reporting the execution time of the training process.
 *
 * Process Steps:
 * 1. If the duration is less than 1000 milliseconds, it returns the duration in milliseconds.
 * 2. For durations longer than 1000 milliseconds, it converts and returns the duration in the largest unit that
 *    does not exceed the duration.
 *
 * Parameters:
 * - `milliseconds`: The time duration in milliseconds.
 *
 * Returns:
 * - A string representation of the duration in the most appropriate unit, formatted with the unit's name.
 *
 * Limitations and Conditions:
 * - The function provides a linear conversion through time units greater than weeks like months and years.
 */
func formatTime(milliseconds float64) string
{
   var seconds, minutes, hours, days, weeks float64
   
   if (milliseconds < 1000.0)
   {
      return fmt.Sprintf("%f milliseconds", milliseconds)
   }
   
   seconds = milliseconds / 1000.0
   if (seconds < 60.0)
   {
      return fmt.Sprintf("%f seconds", seconds)
   }
   
   minutes = seconds / 60.0
   if (minutes < 60.0)
   {
      return fmt.Sprintf("%f minutes", minutes)
   }
   
   hours = minutes / 60.0
   if (hours < 24.0)
   {
      return fmt.Sprintf("%f hours", hours)
   }
   
   days = hours / 24.0
   if (days < 7.0)
   {
      return fmt.Sprintf("%f days", days)
   }
   
   weeks = days / 7.0
   return fmt.Sprintf("%f weeks", weeks)
} // func formatTime(milliseconds float64) string
