/**
 * This module creates types and functions for the purpose of defining an A-B-C-D neural network. The module includes defining
 * structures that encase network parameters (learning rate, number of shallow hidden nodes, number of deep hidden nodes, number
 * of input nodes, number of output nodes, number of test cases, writing weights to a file, the weights file name,
 * training mode, weight initialization, random weight generation lower and upper bounds, error threshold to
 * stop training, and max iterations for training). Furthermore, the NetworkArrays struct defines the arrays and weights following
 * the allocation of these structures.
 * The main function sets the network parameters using 'loadNetworkParameters()', echos the parameters through console using
 * 'echoNetworkParameters()', allocates the memory needed for the network arrays using 'allocateNetworkMemory()', populates
 * the network arrays by initializing the weights using 'populateNetworkMemory()', trains the network using 'train()', and
 * runs the network using 'runTrain()' and 'run()' for every row of the truth table.
 * This module includes functions that train and run the network for a given truth table, as well as "setup" functions that
 * create, allocate, and populate the A-B-C-D neural network. The training algorithm utilizes gradient descent to minimize the
 * error function. The network also uses backpropagation to update the weights efficiently.
 * @author Daniel Gergov
 * @creation 3/1/24
 */

package main

import
(
   "fmt"     // import the io interface
   "math"    // import the math library for max integer
   "math/rand"
   "time"    // import time to measure the execution time
   "os"      // allows interaction with os
   "errors"  // allows for error handling
   "strconv" // convert int to string
   "bufio"   // read and write files
   "strings" // manipulate strings

   "github.com/spf13/viper" // viper for config parsing
) // import

const MILLISECONDS_IN_SECOND float64 = 1000.0
const SECONDS_IN_MINUTE float64 = 60.0
const MINUTES_IN_HOUR float64 = 60.0
const HOURS_IN_DAY float64 = 24.0
const DAYS_IN_WEEK float64 = 7.0

const NUM_BITS_IN_FLOAT64 int = 64

const CONFIG_FILE string = "config.txt"
const CONFIG_FROM_FILE bool = true

const INPUT_OUTPUT_ACTIVATIONS int = 2
const TWO_HIDDEN_LAYERS int = 2


type NetworkParameters struct
{
   learningRate float64
   
   numHiddenLayers int
   numInputNodes   int
   numShallowHiddenNodes  int
   numDeepHiddenNodes  int
   numOutputNodes  int

   numTestCases     int
   externalTestData bool
   testDataFile     string
   
   trainMode          bool
   weightInit         int // 1 is randomize, 2 is zero, 3 is manual, 4 is load from file
   writeWeights       bool
   fileName           string
   activationFunction string // sigmoid, tanh, linear
   
   weightLowerBound float64
   weightUpperBound float64
   
   errorThreshold  float64
   maxIterations   int
   weightSaveEvery int
   keepAliveEvery  int
} // type NetworkParameters struct

type NetworkArrays struct
{
   activations              *[][]float64
   weights                  *[][][]float64
   thetas                   *[][]float64
   psis                     *[][]float64
} // type NetworkArrays struct

var parameters NetworkParameters
var arrays NetworkArrays
var truthTable [][]float64
var expectedOutputs [][]float64
var testedOutputs [][]float64
var numActivationArray []int

var trainStart time.Time
var done bool
var epochError float64
var inputError float64
var epoch int
var executionTime float64

var networkDepth int // input + hidden layers + output
var mLayer, kLayer, jLayer, outputLayer int // layer index

var activationFunction func(float64) float64
var activationPrime func(float64) float64

/**
 * The main function initiates network parameter configuration, memory allocation for network operations, and executes
 * network training and testing. It follows these steps:
 * 1. Sets and displays network parameters. Calls `setNetworkParameters` or `loadNetworkParameters` based on the
 *    `CONFIG_FROM_FILE` constant.
 * 2. Allocates and populates network memory for arrays, truth table, and expected outputs.
 * 3. Trains the network with provided truth table and expected outputs if the trainMode network configuration is 'true'
 * 4. Reports results by iterating over the truth table, comparing and outputting expected outputs
 *    against predictions for each input.
 * 5. Saves the weights to a file if the writeWeights network configuration is 'true'.
 *
 * Limitations:
 * - Assumes correct implementation of and depends on external functions (setNetworkParameters, echoNetworkParameters,
 *    allocateNetworkMemory, populateNetworkMemory, train, run) without error handling.
 * - Requires that truthTable and expectedOutputs are pre-declared and their sizes match, ensuring each input
 *    correlates with its expected output.
 */
func main()
{
   if (CONFIG_FROM_FILE)
   {
      loadNetworkParameters()
   }
   else
   {
      setNetworkParameters()
   }

   echoNetworkParameters()
   
   arrays, truthTable, expectedOutputs, testedOutputs = allocateNetworkMemory()
   populateNetworkMemory()
   
   if (parameters.trainMode)
   {
      train(truthTable, expectedOutputs)
   }
   
   testNetwork()
   reportResults()

   if (parameters.writeWeights)
   {
      saveWeights()
   }
} // func main()

/**
 * The loadNetworkParameters function reads the network configuration parameters from a file and sets the global `parameters`
 * structure accordingly. It then sets the corresponding parameter in the `parameters` structure.
 *
 * The config file is assumed to have all necessary parameters set with parameter name and a space-separated value.
 * The function uses the viper package to read the configuration file and set the parameters in the `parameters` structure.
 *
 * Process:
 * 1. The function checks if the configuration file exists. If not, the function panics.
 * 2. The function uses the viper package to read the configuration file and set the parameters in the `parameters` structure.
 *
 * Syntax:
 * - os.Stat(filename string) checks if the file exists.
 *
 * Limitations:
 * - Assumes that the configuration file exists and is correctly formatted.
 * - Assumes that the configuration file contains all necessary parameters and that the values are correctly formatted.
 */
func loadNetworkParameters()
{
   var err error
   var fileExists bool = false

   _, err = os.Stat(CONFIG_FILE)
   if (err == nil)
   {
      fileExists = true
   }

   if (!fileExists)
   {
      panic("Configuration file does not exist!")
   }

   customConfiguration(CONFIG_FILE)

   parameters.learningRate = viper.GetFloat64("learningRate")
   parameters.numHiddenLayers = viper.GetInt("numHiddenLayers")
   parameters.numInputNodes = viper.GetInt("numInputNodes")
   parameters.numShallowHiddenNodes = viper.GetInt("numShallowHiddenNodes")
   parameters.numDeepHiddenNodes = viper.GetInt("numDeepHiddenNodes")
   parameters.numOutputNodes = viper.GetInt("numOutputNodes")
   parameters.numTestCases = viper.GetInt("numTestCases")
   parameters.externalTestData = viper.GetBool("externalTestData")
   parameters.testDataFile = viper.GetString("testDataFile")
   parameters.trainMode = viper.GetBool("trainMode")
   parameters.weightInit = viper.GetInt("weightInit")
   parameters.writeWeights = viper.GetBool("writeWeights")
   parameters.fileName = viper.GetString("fileName")
   parameters.activationFunction = viper.GetString("activationFunction")
   parameters.weightLowerBound = viper.GetFloat64("weightLowerBound")
   parameters.weightUpperBound = viper.GetFloat64("weightUpperBound")
   parameters.errorThreshold = viper.GetFloat64("errorThreshold")
   parameters.maxIterations = viper.GetInt("maxIterations")
   parameters.weightSaveEvery = viper.GetInt("weightSaveEvery")
   parameters.keepAliveEvery = viper.GetInt("keepAliveEvery")
} // func loadNetworkParameters()

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
 * - externalTestData: Boolean indicating if the test data is external to the program.
 * - testDataFile: The name of the file containing the test data.
 * - trainMode: Boolean indicating if the network is in training mode.
 * - activationFunction: The activation function to be used by the network.
 * - weightInit: Method or value for weight initialization; 1 being random, 2 being zeroes, 3 being manual, 4 being load from file.
 * - weightLowerBound, weightUpperBound: Define the range of values for initializing the network weights to random values.
 * - errorThreshold: The error level at which training is considered sufficiently complete.
 * - maxIterations: Limits the number of training iterations.
 * - writeWeights: Boolean indicating if the weights should be written to a file.
 * - fileName: The name of the file to write the weights to.
 * - weightSaveEvery: The number of iterations to save the weights after.
 * - keepAliveEvery: The number of iterations to output a keep-alive message.
 *
 * Limitations:
 * - The function statically sets network parameters without accepting external input.
 */
func setNetworkParameters()
{
   parameters = NetworkParameters
   {
      learningRate:          0.3,
      numHiddenLayers:       2,
      numInputNodes:         2,
      numShallowHiddenNodes: 5,
      numDeepHiddenNodes:    5,
      numOutputNodes:        3,
      numTestCases:          4,
      externalTestData:      true,
      testDataFile:          "data.txt",
      trainMode:             true,
      weightInit:            1,
      writeWeights:          true,
      fileName:              "weights.txt",
      activationFunction:    "sigmoid",
      weightLowerBound:      0.1,
      weightUpperBound:      1.5,
      errorThreshold:        2e-4,
      maxIterations:         100000,
      weightSaveEvery:       10000,
      keepAliveEvery:        100000,
   }
} // func setNetworkParameters()

/**
 * The echoNetworkParameters function prints the current network configuration parameters to the console. The function clearly
 * displays the network's configurations including learning rate (λ), error threshold, maximum number of iterations,
 * network architecture (input-hidden-output node counts), number of test cases, training mode status, weight initialization
 * method, and the range for random weight initialization.
 *
 * Displayed Parameters:
 * - Learning rate (λ), error threshold, and maximum iterations for training control only if train mode is true.
 * - Network architecture detailed by the count of input, shallow hidden, deep hidden, and output nodes.
 * - Activation function used by the network.
 * - The total number of test cases used for training or validation.
 * - Training mode indicator (true for training mode).
 * - Test data source indicator (true for external test data).
 * - Weight initialization method, where "1" denotes random initialization and "2" denotes initialization to zero, 3 denotes
 * -    manual initialization, and 4 denotes loading from a file.
 * - Keep alive and save weights every N iterations.
 * - Range for random weight initialization, specifying lower and upper bounds.
 *
 * Limitations:
 * - This function depends on the global `parameters` structure being set by setNetworkParameters() and not by passed arguments.
 */
func echoNetworkParameters()
{
   fmt.Println("Network Parameters:")
   fmt.Println("-------------------")

   if (parameters.trainMode)
   {
      fmt.Printf("λ: %v, Error Threshold: %v, Max Iterations: %d\n",
                 parameters.learningRate, parameters.errorThreshold, parameters.maxIterations)
   }

   fmt.Printf("Network: %d-%d-%d-%d, NumberTestCases: %d\n",
      parameters.numInputNodes, parameters.numShallowHiddenNodes, parameters.numDeepHiddenNodes, parameters.numOutputNodes,
      parameters.numTestCases)

   fmt.Printf("Activation Function: %s\n", parameters.activationFunction)
   fmt.Printf("Train Mode: %t\n", parameters.trainMode)
   fmt.Printf("Weight Init: %d -- 1 = random, 2 = zero, 3 = manual, 4 = load from file\n", parameters.weightInit)
   fmt.Printf("Test Data: %t -- true = external, 2 = internal\n", parameters.externalTestData)
   fmt.Printf("Keep Alive Every: %d, Save Weights Every: %d\n", parameters.keepAliveEvery, parameters.weightSaveEvery)
   fmt.Printf("Random Range [%v, %v]\n\n", parameters.weightLowerBound, parameters.weightUpperBound)
} // func echoNetworkParameters()

/**
 * The allocateNetworkMemory function is responsible for initializing and allocating memory for various arrays and matrices used
 * by the network, including those for input to shallow hidden layer weights, shallow hidden to deep hidden layer weights,
 * deep hidden to output layer weights, and other structures for training such as thetas, and psis.
 * The function also allocates the truth table for inputs and expected outputs.

 * Returns:
 * - A NetworkArrays structure containing references to all allocated arrays and matrices used by the network.
 * - A truth table for network inputs as a slice of float64 slices.
 * - An output truth table as a slice of float64.
 * - A ran outputs table as a slice of float64.

 * If the trainMode parameter is true, structures used exclusively for training (thetas, psis)
 * are allocated. This condition helps optimize memory usage by only allocating necessary arrays.

 * Limitations:
 * - Assumes that the global `parameters` structure is correctly initialized before this function is called.
 */
func allocateNetworkMemory() (NetworkArrays, [][]float64, [][]float64, [][]float64)
{
   if (parameters.numHiddenLayers != TWO_HIDDEN_LAYERS)
   {
      panic("2 hidden layers are only supported!")
   } // if (parameters.numHiddenLayers != 2)

   var alpha, beta, input, output int
   
   networkDepth = parameters.numHiddenLayers + INPUT_OUTPUT_ACTIVATIONS
   outputLayer = networkDepth - 1
   jLayer = outputLayer - 1
   kLayer = jLayer - 1
   mLayer = kLayer - 1
   
   numActivationArray = []int{parameters.numInputNodes, parameters.numShallowHiddenNodes, parameters.numDeepHiddenNodes,
                              parameters.numOutputNodes}

   var activations [][]float64 = make([][]float64, networkDepth)
   for alpha = range activations
   {
      activations[alpha] = make([]float64, numActivationArray[alpha])
   } // for alpha = range activations


   var weights [][][]float64 = make([][][]float64, networkDepth - 1)
   for alpha = 0; alpha < networkDepth - 1; alpha++
   {
      weights[alpha] = make([][]float64, numActivationArray[alpha])

      for beta = 0; beta < numActivationArray[alpha]; beta++
      {
         weights[alpha][beta] = make([]float64, numActivationArray[alpha + 1])
      } // for beta = 0; beta < numActivationArray[alpha]; beta++
   } // for alpha = 0; alpha < networkDepth - 1; alpha++
   
   var thetas, psis [][]float64
   
   if (parameters.trainMode)
   {
      thetas = make([][]float64, networkDepth - 1)
      for alpha = 1; alpha < networkDepth - 1; alpha++
      {
         thetas[alpha] = make([]float64, numActivationArray[alpha])
      } // for alpha = 1; alpha < networkDepth - 1; alpha++

      psis = make([][]float64, networkDepth)
      for alpha = 1; alpha < networkDepth; alpha++
      {
         psis[alpha] = make([]float64, numActivationArray[alpha])
      } // for alpha = 1; alpha < networkDepth; alpha++
   } // if (parameters.trainMode)
   
   var inputTruthTable [][]float64 = make([][]float64, parameters.numTestCases)
   for input = range inputTruthTable
   {
      inputTruthTable[input] = make([]float64, parameters.numInputNodes)
   } // for input = range inputTruthTable
   
   var outputTruthTable [][]float64 = make([][]float64, parameters.numTestCases)
   for output = range outputTruthTable
   {
      outputTruthTable[output] = make([]float64, parameters.numOutputNodes)
   } // for output = range outputTruthTable

   var ranOutputs [][]float64 = make([][]float64, parameters.numTestCases)
   for output = range ranOutputs
   {
      ranOutputs[output] = make([]float64, parameters.numOutputNodes)
   } // for output = range ranOutputs
   
   return NetworkArrays
   {
      activations:                    &activations,
      weights:                        &weights,
      thetas:                         &thetas,
      psis:                           &psis,
   }, inputTruthTable, outputTruthTable, ranOutputs
} // func allocateNetworkMemory() (NetworkArrays, [][]float64, [][]float64, [][]float64)

/**
 * The loadTestData function reads the test data from a file and populates the truth table and expected outputs for the network.
 * The function reads the test data file line by line, parsing each line to extract the input and expected output values.
 * It then sets the corresponding values in the `truthTable` and `expectedOutputs` arrays.
 *
 * Syntax:
 * - os.Stat(filename string) checks if the file exists.
 * - os.OpenFile(filename string, flag int, perm os.FileMode) opens a file for writing.
 * - error.Is(err error, target error) checks if the error is equal to the target error.
 * - defer file.Close() defers the file's closure until the function returns.
 * - bufio.NewScanner(file *os.File) creates a new scanner to read the file line by line.
 * - scanner.Scan() reads the next line from the file.
 * - scanner.Text() returns the current line from the scanner.
 * - strings.Fields(s string) splits the string into fields separated by whitespace.
 * - strconv.ParseFloat(s string, bitSize int) converts a string to a float64.
 *
 * Limitations:
 * - Assumes that the test data file exists and is correctly formatted.
 * - Assumes the `truthTable` and `expectedOutput` arrays are properly allocated.
 */
func loadTestData()
{
   var file *os.File
   var err error
   var fileExists bool = false
   var testLine string
   var test, m, i int
   var parts []string

   _, err = os.Stat(parameters.testDataFile)
   if (err == nil)
   {
      fileExists = true
   }

   if (!fileExists)
   {
      panic("Test data file does not exist!")
   }

   file, err = os.OpenFile(parameters.testDataFile, os.O_RDONLY, 0644) // open file in read-only mode
   checkError(err)
 
   defer file.Close()
 
   var scanner *bufio.Scanner = bufio.NewScanner(file)

   test = 0
 
   for (scanner.Scan() && test < parameters.numTestCases)
   {
      testLine = scanner.Text()
      parts = strings.Fields(testLine)
      if (len(parts) == parameters.numInputNodes + parameters.numOutputNodes + 1)
      {
         for m = 0; m < parameters.numInputNodes; m++
         {
            truthTable[test][m], _ = strconv.ParseFloat(parts[m], NUM_BITS_IN_FLOAT64)
         }
         for i = 0; i < parameters.numOutputNodes; i++
         {
            expectedOutputs[test][i], _ = strconv.ParseFloat(parts[i + parameters.numInputNodes + 1], NUM_BITS_IN_FLOAT64)
         }
      } // if (len(parts) == parameters.numInputNodes + parameters.numOutputNodes + 1)
      else
      {
         panic("Test data file is not formatted correctly!")
      } // else
      test++
   } // for (scanner.Scan() && test < parameters.numTestCases)

   if (test != parameters.numTestCases)
   {
      panic("Test data file does not contain the correct amount of test cases!")
   }
} // func loadTestData()

/**
 * The populateNetworkMemory function initializes the network's weight matrices and sets up the truth table and expected outputs
 * for training or evaluation. It follows these steps:
 *
 * 1. If the weight initialization mode is set to random (weightInit == 1), it initializes the input-hidden and hidden-output
 * weight matrices with random values within the bounds (weightLowerBound to weightUpperBound).
 * 2. If the weight initialization mode is set to manual (weightInit == 3), it initializes the input-hidden and hidden-output
 *    weight matrices with predefined values for a 2-2-1 network.
 * 3. If the weight initialization mode is set to load from file (weightInit == 4), it loads the weights from a file.
 * 4. Populates the truth table with predefined inputs.
 * 5. Sets the expected outputs corresponding to the truth table inputs to a binary operation either XOR, OR, or AND.
 *
 * Limitations:
 * - Assumes `arrays`, `truthTable`, and `expectedOutputs` are globally accessible and correctly linked to the network's structure.
 */
func populateNetworkMemory()
{
   var alpha, beta, betaNought int
   
   if (parameters.weightInit == 1)
   {
      rand.Seed(time.Now().UnixNano())
      for alpha = 0; alpha < networkDepth - 1; alpha++
      {
         for beta = 0; beta < numActivationArray[alpha]; beta++
         {
            for betaNought = 0; betaNought < numActivationArray[alpha + 1]; betaNought++
            {
               (*arrays.weights)[alpha][beta][betaNought] = randomNumber(parameters.weightLowerBound, parameters.weightUpperBound)
            } // for betaNought = 0; betaNought < numActivationArray[alpha + 1]; betaNought++
         } // for beta = 0; beta < numActivationArray[alpha]; beta++
      } // for alpha = 0; alpha < networkDepth - 1; alpha++
   } // if (parameters.weightInit == 1)
   else if (parameters.weightInit == 3)
   {
      (*arrays.weights)[0][0][0] = 0.8
      (*arrays.weights)[0][0][1] = 0.5
      (*arrays.weights)[0][1][0] = 0.5
      (*arrays.weights)[0][1][1] = 0.5

      (*arrays.weights)[1][0][0] = -0.5
      (*arrays.weights)[1][1][0] = 0.5
   } // else if (parameters.weightInit == 3)
   else if (parameters.weightInit == 4)
   {
      loadWeights()
   } // else if (parameters.weightInit == 4)

   if (parameters.externalTestData)
   {
      loadTestData()
   } // if (parameters.externalTestData)
   else
   {
      truthTable[0][0] = 0.0
      truthTable[0][1] = 0.0
      // truthTable[0][2] = 0.0
      truthTable[1][0] = 0.0
      truthTable[1][1] = 1.0
      // truthTable[1][2] = 0.0
      truthTable[2][0] = 1.0
      truthTable[2][1] = 0.0
      // truthTable[2][2] = 0.0
      truthTable[3][0] = 1.0
      truthTable[3][1] = 1.0
      // truthTable[3][2] = 0.0
      
      expectedOutputs[0][0] = 0.0
      expectedOutputs[0][1] = 0.0
      expectedOutputs[0][2] = 0.0

      expectedOutputs[1][0] = 0.0
      expectedOutputs[1][1] = 1.0
      expectedOutputs[1][2] = 1.0

      expectedOutputs[2][0] = 0.0
      expectedOutputs[2][1] = 1.0
      expectedOutputs[2][2] = 1.0

      expectedOutputs[3][0] = 1.0
      expectedOutputs[3][1] = 1.0
      expectedOutputs[3][2] = 0.0
   } // if (parameters.externalTestData)

   assignActivationFunction()
   assignActivationPrime()
} // func populateNetworkMemory()

/**
 * The randomNumber function generates a random floating-point number within a given range. The function uses the `rand` package.
 * The function is used to initialize the network's weights with random values within a specified range.
 */
func randomNumber(lowerBound, upperBound float64) float64
{
   return lowerBound + rand.Float64() * (upperBound - lowerBound)
} // func randomNumber(lowerBound, upperBound float64) float64

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
 * The hyperbolic tangent function calculates the tanh activation of a given input value `x`. The tanh activation function
 * introduces non-linearity into a model.
 *
 * The tanh of `x` follows the formula:
 * tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *
 * Parameters:
 * - x: The input value for which to apply tanh formula.
 *
 * Returns:
 * - The tanh activation of `x`, a float64 value in the range (-1, 1).
 */
func tanh(x float64) float64
{
   return math.Tanh(x)
}

/**
 * The tanhDerivative function computes the derivative of the tanh activation function for a given input value `x`.
 * The derivative is used the calculation for the delta weights through computing the psi variables. The function first
 * calculates the tanh of `x`, then applies the derivative formula of the tanh function.
 *
 * The derivative of the tanh function S(x) is given by:
 * T'(x) = 1 - (T(x) - T(x))
 * where T(x) is the tanh of `x`.
 *
 * Parameters:
 * - x: The input value for which to apply the tanh derivative formula.
 *
 * Returns:
 * - The derivative of the tanh function of `x`, a float64 value in the range (0, 1].
 */
func tanhDerivative(x float64) float64
{
   x = tanh(x)
   return 1.0 - (x * x)
}

/**
 * The linear function returns the given input value `x`. The linear activation function reverts the model back to linearity.
 *
 * The linear activation function of `x` follows the formula:
 * linear(x) = x
 *
 * Parameters:
 * - x: The input value for which to apply linear function.
 *
 * Returns:
 * - `x`, a float64 value which is the same as the input `x`.
 */
func linear(x float64) float64
{
   return x
}

/**
 * The linearDerivative function computes the derivative of the linear activation function for a given input value `x`.
 * The derivative is used the calculation for the delta weights through computing the psi variables. The function returns 1.
 *
 * The derivative of the linear function L(x) is given by:
 * L'(x) = 1
 *
 * Parameters:
 * - x: The input value for which to apply the linear derivative function.
 *
 * Returns:
 * - The value 1.
 */
func linearDerivative(x float64) float64
{
   return 1.0
}

/**
 * The assignActivationFunction function acts as a function to set the activation function being used.
 * The function allows the ability to change the activation function using a config file in other functions. The function assigns
 * the activation function to the function specified in the config file.
 */
func assignActivationFunction()
{
   if (parameters.activationFunction == "sigmoid")
   {
      activationFunction = sigmoid
   }
   else
   {
      if (parameters.activationFunction == "tanh")
      {
         activationFunction = tanh
      }
      else
      {
         activationFunction = linear
      }
   } // if (parameters.activationFunction == "sigmoid")
} // func assignActivationFunction()

/**
 * The assignActivationPrime function acts as a function to set the activation function's derivative being used.
 * The function allows the ability to change the activation function using a config file in other functions. The function assigns
 * the activation function's derivative to the function specified in the config file.
 */
func assignActivationPrime()
{
   if (parameters.activationFunction == "sigmoid")
   {
      activationPrime = sigmoidDerivative
   }
   else
   {
      if (parameters.activationFunction == "tanh")
      {
         activationPrime = tanhDerivative
      }
      else
      {
         activationPrime = linearDerivative
      }
   } // if (parameters.activationFunction == "sigmoid")
} // func assignActivationPrime()

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
 *    - Executes backwardpropagation to calculate and update the delta weights for every connectivity layer.
 *    - Performs another forward propagation to update the error
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
func train(inputs [][]float64, expectedOutputs [][]float64)
{
   trainStart = time.Now()
   done = false
   
   epochError = math.MaxFloat64
   epoch = 0

   var input, m, k, j, i int
   var omegaK, omegaJ, omegaI float64
   
   var activations [][]float64 = *arrays.activations

   var thetas [][]float64 = *arrays.thetas
   var psis [][]float64 = *arrays.psis

   var weights [][][]float64 = *arrays.weights
   
   for (!done)
   {
      epochError = 0.0
      for input = 0; input < parameters.numTestCases; input++
      {
         inputError = 0.0

         runTrain(&activations, &thetas, &psis, &weights, &inputs[input], &expectedOutputs[input])

         for j = 0; j < parameters.numDeepHiddenNodes; j++
         {
            omegaJ = 0.0
            for i = 0; i < parameters.numOutputNodes; i++
            {
               omegaJ += psis[outputLayer][i] * weights[jLayer][j][i]
               weights[jLayer][j][i] += parameters.learningRate * activations[jLayer][j] * psis[outputLayer][i]
            } // for i = 0; i < parameters.numOutputNodes; i++

            psis[jLayer][j] = omegaJ * activationPrime(thetas[jLayer][j])
         } // for j = 0; j < parameters.numDeepHiddenNodes; j++

         for k = 0; k < parameters.numShallowHiddenNodes; k++
         {
            omegaK = 0.0
            for j = 0; j < parameters.numDeepHiddenNodes; j++
            {
               omegaK += psis[jLayer][j] * weights[kLayer][k][j]
               weights[kLayer][k][j] += parameters.learningRate * activations[kLayer][k] * psis[jLayer][j]
            } // for j = 0; j < parameters.numDeepHiddenNodes; j++

            psis[kLayer][k] = omegaK * activationPrime(thetas[kLayer][k])

            for m = 0; m < parameters.numInputNodes; m++
            {
               weights[mLayer][m][k] += parameters.learningRate * activations[mLayer][m] * psis[kLayer][k]
            } // for m = 0; m < parameters.numInputNodes; m++
         } // for k = 0; k < parameters.numShallowHiddenNodes; k++

         for i = range run(inputs[input])
         {
            omegaI = expectedOutputs[input][i] - activations[outputLayer][i]
            inputError += 0.5 * omegaI * omegaI
         } // for i = range run(inputs[input])

         epochError += inputError
      } // for input = 0; input < parameters.numTestCases; input++

      epochError /= float64(parameters.numTestCases)
      epoch++
      done = epochError < parameters.errorThreshold || epoch > parameters.maxIterations

      if (epoch % parameters.weightSaveEvery == 0)
      {
         saveWeights()
         fmt.Println("Weights saved...")
      } // if (epoch % parameters.weightSaveEvery == 0)

      if (epoch % parameters.keepAliveEvery == 0)
      {
         fmt.Printf("Finished epoch %d with error %f\n", epoch, epochError)
      } // if (epoch % parameters.keepAliveEvery == 0)
   } // for (!done)
   
   executionTime = float64(time.Since(trainStart) / time.Millisecond)
} // func train(inputs [][]float64, expectedOutputs [][]float64)

/**
 * The testNetwork function runs the network for each input in the truth table, storing the network's predictions in the
 * `testedOutputs` array. The function is used to test the network's performance after training.
 *
 * Limitations:
 * - Assumes that the global `truthTable` and `expectedOutputs` arrays are correctly initialized and match in size.
 */
func testNetwork()
{
   var index, inner int
   var input []float64
   var num float64
   
   for index, input = range truthTable
   {
      for inner, num = range (run(input))
      {
         testedOutputs[index][inner] = num
      }
   } // for index, input = range truthTable
} // func testNetwork()

/**
 * The reportError function prints the current training error and the number of iterations to the console. The function is
 * used to report the training progress and the reason for stopping the training process. It is called after the training
 * process is completed.
 *
 * Limitations:
 * - Assumes that the global `epochError` and `epoch` variables are correctly updated during the training process.
 */
func reportResults()
{
   if (parameters.trainMode)
   {
      var formattedTime string = formatTime(executionTime)
      fmt.Printf("Training stopped after %s and %v iterations with average error: %.9f.\n", formattedTime, epoch, epochError)
      fmt.Printf("Reason for stopping: ")

      if (epoch >= parameters.maxIterations)
      {
         fmt.Printf("Exceeded max iterations of %d\n", parameters.maxIterations)
      }
      
      if (epochError <= parameters.errorThreshold)
      {
         fmt.Printf("Error became less than error threshold %.7f\n", epochError)
      }
   } // if (parameters.trainMode)

   var input []float64
   var index int

   for index, input = range truthTable
   {
      fmt.Printf("Input: %v, Expected: %f, Predicted: %f\n", input, expectedOutputs[index], testedOutputs[index])
   }
} // func reportResults()

/**
 * The run function performs forward propagation through the network for a given input array `a`, computing the network's output.
 * The function uses the input array to compute dot products between the input neurons and the weights, eventually computing the
 * value of the output node (the network's output).
 *
 * Process Overview:
 * 1. Computes weighted sums at the shallow hidden nodes, applying the sigmoid activation function to each sum/theta.
 * 2. Collects the shallow hidden activations, again applying weights and the sigmoid function to get the deep hidden activations.
 * 2. Collects the deep hidden activations, again applying weights and the sigmoid function to produce the final output array.
 * 3. Returns the network's predictions for the given input.
 *
 * Parameters:
 * - `a`: Input array to use to make a prediction.
 *
 * Limitations:
 * - Assumes that the network's weights (`weights`) have been properly initialized.
 * - Assumes that the input array `a` matches the size expected by the network.
 */
func run(a []float64) []float64
{
   var m, k, j, i int
   
   var activations [][]float64 = *arrays.activations
   var weights [][][]float64 = *arrays.weights
   
   for k = 0; k < parameters.numShallowHiddenNodes; k++
   {
      var sum float64 = 0.0
      for m = 0; m < parameters.numInputNodes; m++
      {
         sum += a[m] * weights[mLayer][m][k]
      }
      activations[kLayer][k] = activationFunction(sum)
   } // for k = 0; k < parameters.numShallowHiddenNodes; k++
   
   for j = 0; j < parameters.numDeepHiddenNodes; j++
   {
      var sum float64 = 0.0
      for k = 0; k < parameters.numShallowHiddenNodes; k++
      {
         sum += activations[kLayer][k] * weights[kLayer][k][j]
      }
      activations[jLayer][j] = activationFunction(sum)
   } // for j = 0; j < parameters.numDeepHiddenNodes; j++

   for i = 0; i < parameters.numOutputNodes; i++
   {
      var sum float64 = 0.0
      for j = 0; j < parameters.numDeepHiddenNodes; j++
      {
         sum += activations[jLayer][j] * weights[jLayer][j][i]
      }
      activations[outputLayer][i] = activationFunction(sum)
   } // for i = 0; i < parameters.numOutputNodes; i++
   
   return (*arrays.activations)[outputLayer]
} // func run(a []float64) []float64

/**
 * The runTrain function is designed for the forward propagation part of the neural network training process.
 * It updates the network's hidden layer outputs and the final output array (F) value based on a given input array.
 *
 * Process:
 *    1. Copies the input vector into the network's input layer.
 *    2. For each hidden neuron, computes the weighted sum of its inputs and applies the sigmoid function to
 *       obtain the neuron's output.
 *    3. Calculates the weighted sum of each hidden layer outputs and applies the sigmoid function to
 *       determine the network outputs.
 *    4. Computes the psis array used for network training.
 *
 * Parameters:
 * - `activations`: Reference to the all the network's activations.
 * - `thetas`: Reference to the variable storing the weighted sum of right-side layer's outputs before applying sigmoid.
 * - `psis`: Reference to the vector storing the error gradient for the output layer for every right-side layer.
 * - `weights`: Reference to the vector of matrix of weights for every connectivity layer.
 * - `input`: The input vector for the current training example.
 * - `outputs`: The expected outputs for the current training example.
 *
 * Limitations and Conditions:
 * - Assumes that the network's weights (`weights`) have been properly initialized.
 */
func runTrain(activations *[][]float64, thetas *[][]float64, psis *[][]float64, weights *[][][]float64,
              input *[]float64, outputs *[]float64)
{
   var m, k, j, i int
   var omegaI, thetaI float64
   
   for m = 0; m < parameters.numInputNodes; m++
   {
      (*activations)[mLayer][m] = (*input)[m]
   } // for m = 0; m < parameters.numInputNodes; m++
   
   for k = 0; k < parameters.numShallowHiddenNodes; k++
   {
      (*thetas)[kLayer][k] = 0.0
      for m = 0; m < parameters.numInputNodes; m++
      {
         (*thetas)[kLayer][k] += (*activations)[mLayer][m] * (*weights)[mLayer][m][k]
      } // for m = 0; m < parameters.numInputNodes; m++
      
      (*activations)[kLayer][k] = activationFunction((*thetas)[kLayer][k])
   } // for k = 0; k < parameters.numShallowHiddenNodes; k++

   for j = 0; j < parameters.numDeepHiddenNodes; j++
   {
      (*thetas)[jLayer][j] = 0.0
      for k = 0; k < parameters.numShallowHiddenNodes; k++
      {
         (*thetas)[jLayer][j] += (*activations)[kLayer][k] * (*weights)[kLayer][k][j]
      } // for k = 0; k < parameters.numShallowHiddenNodes; k++
      
      (*activations)[jLayer][j] = activationFunction((*thetas)[jLayer][j])
   } // for j = 0; j < parameters.numDeepHiddenNodes; j++
   
   for i = 0; i < parameters.numOutputNodes; i++
   {
      thetaI = 0.0
      for j = 0; j < parameters.numDeepHiddenNodes; j++
      {
         thetaI += (*activations)[jLayer][j] * (*weights)[jLayer][j][i]
      } // for j = 0; j < parameters.numDeepHiddenNodes; j++

      (*activations)[outputLayer][i] = activationFunction(thetaI)
      omegaI = (*outputs)[i] - (*activations)[outputLayer][i]
      (*psis)[outputLayer][i] = omegaI * activationPrime(thetaI)
   } // for i = 0; i < parameters.numOutputNodes; i++
} // func runTrain(activations *[][]float64...

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
   var formatted string
   var override bool = false

   if (milliseconds < MILLISECONDS_IN_SECOND)
   {
      formatted = fmt.Sprintf("%f milliseconds", milliseconds)
      override = true
   }
   
   seconds = milliseconds / MILLISECONDS_IN_SECOND
   if (seconds < SECONDS_IN_MINUTE && !override)
   {
      formatted = fmt.Sprintf("%f seconds", seconds)
      override = true
   }
   
   minutes = seconds / SECONDS_IN_MINUTE
   if (minutes < MINUTES_IN_HOUR && !override)
   {
      formatted = fmt.Sprintf("%f minutes", minutes)
      override = true
   }
   
   hours = minutes / MINUTES_IN_HOUR
   if (hours < HOURS_IN_DAY && !override)
   {
      formatted = fmt.Sprintf("%f hours", hours)
      override = true
   }
   
   days = hours / HOURS_IN_DAY
   if (days < DAYS_IN_WEEK && !override)
   {
      formatted = fmt.Sprintf("%f days", days)
      override = true
   }

   if (days >= DAYS_IN_WEEK && !override)
   {
      weeks = days / DAYS_IN_WEEK
      formatted = fmt.Sprintf("%f weeks", weeks)
   }

   return formatted
} // func formatTime(milliseconds float64) string

/**
 * The checkError function is a utility function that checks if an error is not nil and panics if it is. It is used to
 * check for errors in the file I/O operations.
 *
 * Parameters:
 * - `err`: The error to check.
 *
 * Returns:
 * - The function does not return a value but panics if there is an error.
 */
func checkError(err error)
{
   if (err != nil)
   {
      fmt.Println("Received an error:", err)
      panic(err)
   } // if (err != nil)
} // func checkError(err error)

/**
 * The saveWeights function writes the network's weights to a file. The function opens a file for writing and writes the
 * network's architecture and weights to the file. The architecture is written as a string in the
 * format "input-hidden-hidden-output".
 * The input-shallow_hidden weights are written first, followed by the shallow_hidden-deep_hidden weights, followed by the
 * deep_hidden-output weights.
 *
 * Syntax:
 * - os.Stat(filename string) checks if the file exists.
 * - os.Create(filename string) creates a new file.
 * - os.OpenFile(filename string, flag int, perm os.FileMode) opens a file for writing.
 * - file.WriteString(s string) writes a string to the file.
 * - error.Is(err error, target error) checks if the error is equal to the target error.
 * - defer file.Close() defers the file's closure until the function returns.
 * - os.Truncate(filename string, 0) clears the contents of a file.
 *
 * Process:
 * 1. Checks if the file exists and creates a new file if it does not.
 * 2. Opens the file for writing.
 * 3. Writes the network's architecture to the file.
 * 4. Writes the input-shallow_hidden weights to the file.
 * 5. Writes the shallow_hidden-deep_hidden weights to the file.
 * 6. Writes the deeo_hidden-output weights to the file.
 * 7. Closes the file.
 */
func saveWeights()
{
   var m, k, j, i int
   var file *os.File
   var err error
   var fileExists bool = false

   _, err = os.Stat(parameters.fileName)
   if (err == nil)
   {
      fileExists = true

      err = os.Truncate(parameters.fileName, 0); // clear the file's content
      checkError(err)
   }

   if (!fileExists && errors.Is(err, os.ErrNotExist))
   {
      file, err = os.Create(parameters.fileName) // create the file
      checkError(err)
   }

   file, err = os.OpenFile(parameters.fileName, os.O_WRONLY, 0644) // open the file
   checkError(err)

   defer file.Close()

   _, err = file.WriteString(fmt.Sprintf("%d-%d-%d-%d\n", parameters.numInputNodes, parameters.numShallowHiddenNodes,
                             parameters.numDeepHiddenNodes, parameters.numOutputNodes)) // write network architecture to file
   checkError(err)

   _, err = file.WriteString("\n") // write new line to file
   checkError(err)

   for m = 0; m < parameters.numInputNodes; m++
   {
      for k = 0; k < parameters.numShallowHiddenNodes; k++
      {
         _, err = file.WriteString(fmt.Sprintf("%f\n", (*arrays.weights)[mLayer][m][k]))
         checkError(err)
      } // for k = 0; k < parameters.numShallowHiddenNodes; k++
   } // for m = 0; m < parameters.numInputNodes; m++

   _, err = file.WriteString("\n") // write new line to file
   checkError(err)

   for k = 0; k < parameters.numShallowHiddenNodes; k++
   {
      for j = 0; j < parameters.numDeepHiddenNodes; j++
      {
         _, err = file.WriteString(fmt.Sprintf("%f\n", (*arrays.weights)[kLayer][k][j]))
         checkError(err)
      } // for j = 0; j < parameters.numDeepHiddenNodes; j++
   } // for k = 0; k < parameters.numShallowHiddenNodes; k++

   _, err = file.WriteString("\n") // write new line to file
   checkError(err)

   for j = 0; j < parameters.numDeepHiddenNodes; j++
   {
      for i = 0; i < parameters.numOutputNodes; i++
      {
         _, err = file.WriteString(fmt.Sprintf("%f\n", (*arrays.weights)[jLayer][j][i]))
         checkError(err)
      } // for j = 0; j < parameters.numDeepHiddenNodes; j++
   } // for i = 0; i < parameters.numOutputNodes; i++
} // func saveWeights()

/**
 * The loadWeights function reads the network's weights from a file. The function opens a file for reading and reads the
 * network's architecture and weights from the file. The architecture is read as a string in the
 * format "input-hidden-hidden-output".
 * All of the weights for every connectivity layer are read in order of the network propagation rule. The function then
 * checks if the network's architecture matches the architecture in the weights file before reading in the weights.
 *
 * Syntax:
 * - os.Stat(filename string) checks if the file exists.
 * - os.Open(filename string) opens a file for reading.
 * - fmt.Fscan(file *os.File, a ...interface{}) reads from the file.
 * - error.Is(err error, target error) checks if the error is equal to the target error.
 * - defer file.Close() defers the file's closure until the function returns.
 */
func loadWeights()
{
   var m, k, j, i int
   var file *os.File
   var err error
   var fileExists bool = false

   _, err = os.Stat(parameters.fileName)
   if (!fileExists && errors.Is(err, os.ErrNotExist))
   {
      fmt.Println("File for loading weights does not exist!")
      panic(err)
   }

   file, err = os.OpenFile(parameters.fileName, os.O_RDONLY, 0644) // open the file
   checkError(err)

   defer file.Close()

   var architecture string
   _, err = fmt.Fscan(file, &architecture)
   checkError(err)

   var numInputNodes string = strconv.Itoa(parameters.numInputNodes)
   var numShallowHiddenNodes string = strconv.Itoa(parameters.numShallowHiddenNodes)
   var numDeepHiddenNodes string = strconv.Itoa(parameters.numDeepHiddenNodes)
   var numOutputNodes string = strconv.Itoa(parameters.numOutputNodes)

   var configString string = numInputNodes + "-" + numShallowHiddenNodes + "-" + numDeepHiddenNodes + "-" + numOutputNodes

   if (configString != architecture)
   {
      fmt.Println("Network architecture does not match the architecture in the weights file!")
      panic(err)
   } // if (configString != architecture)

   _, err = fmt.Fscan(file, architecture)

   for m = 0; m < parameters.numInputNodes; m++
   {
      for k = 0; k < parameters.numShallowHiddenNodes; k++
      {
         _, err = fmt.Fscan(file, &(*arrays.weights)[mLayer][m][k])
         checkError(err)
      } // for k = 0; k < parameters.numShallowHiddenNodes; k++
   } // for m = 0; m < parameters.numInputNodes; m++

   _, err = fmt.Fscan(file, architecture)

   for k = 0; k < parameters.numShallowHiddenNodes; k++
   {
      for j = 0; j < parameters.numDeepHiddenNodes; j++
      {
         _, err = fmt.Fscan(file, &(*arrays.weights)[kLayer][k][j])
         checkError(err)
      } // for j = 0; j < parameters.numDeepHiddenNodes; j++
   } // for k = 0; k < parameters.numShallowHiddenNodes; k++

   _, err = fmt.Fscan(file, architecture)

   for j = 0; j < parameters.numDeepHiddenNodes; j++
   {
      for i = 0; i < parameters.numOutputNodes; i++
      {
         _, err = fmt.Fscan(file, &(*arrays.weights)[jLayer][j][i])
         checkError(err)
      } // for i = 0; i < parameters.numOutputNodes; i++
   } // for j = 0; j < parameters.numDeepHiddenNodes; j++
} // func loadWeights()

/**
 * The customConfiguration function defines a custom configuration for the network's parameters. The function reads the
 * configuration from a file and sets the parameters based on the configuration. The configuration file is expected to
 * contain key-value pairs, with each pair separated by a space. The function reads the file line by line, parses each
 * line to extract the key and value, and sets the corresponding parameter in the `viper` configuration.
 *
 * Syntax:
 * - os.Open(filename string) opens a file for reading.
 * - strings.SplitN(s, sep string, n int) splits the string into fields separated by whitespace.
 * - viper.Set(key string, value interface{}) sets the value of a key in the configuration.
 *
 * Parameters:
 * - `filePath`: The path to the configuration file.
 */
func customConfiguration(filePath string)
{
   var file *os.File
   var err error
   var scanner *bufio.Scanner
   var line, key, value string
   var parts []string

   file, err = os.Open(filePath)
   if (err != nil)
   {
      panic(fmt.Errorf("Fatal error in opening config file: %w", err))
   }

   defer file.Close()

   scanner = bufio.NewScanner(file)
   for (scanner.Scan())
   {
      line = scanner.Text()

      if (line == "" || strings.HasPrefix(line, "/") || strings.HasPrefix(line, "|") || strings.HasPrefix(line, "\\"))
      {
         continue
      }

      parts = strings.SplitN(line, " ", 2)
      if (len(parts) == 2)
      {
         key = parts[0]
         value = parts[1]
         viper.Set(key, value)
      } // if (len(parts) == 2)
   } // for (scanner.Scan())

   err = scanner.Err();
   if (err != nil)
   {
      panic(fmt.Errorf("Error in scanning config file: %w", err))
   }
} // func customConfiguration(filePath string)