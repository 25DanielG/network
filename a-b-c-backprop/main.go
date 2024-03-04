/**
 * This module creates types and functions for the purpose of defining an A-B-C neural network. The module includes defining
 * structures that encase network parameters (learning rate, number of hidden nodes, number of input nodes, number of output nodes
 * which is coded to C, number of test cases, training mode, weight initialization, random weight generation lower and
 * upper bounds, error threshold to stop training, writing weights to a file, the weights filename, and max iterations
 * for training). Furthermore, the NetworkArrays struct defines the arrays and weights following
 * the allocation of these structures.
 * The main function sets the network parameters using 'loadNetworkParameters()', echos the parameters through console using
 * 'echoNetworkParameters()', allocates the memory needed for the network arrays using 'allocateNetworkMemory()', populates
 * the network arrays by initializing the weights using 'populateNetworkMemory()', trains the network using 'train()', and
 * runs the network using 'runTrain()' and 'run()' for every row of the truth table.
 * This module includes functions that train and run the network for a given truth table, as well as "setup" functions that
 * create, allocate, and populate the A-B-C neural network. The training algorithm utilizes gradient descent to minimize the
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
) // import

const MILLISECONDS_IN_SECOND float64 = 1000.0
const SECONDS_IN_MINUTE float64 = 60.0
const MINUTES_IN_HOUR float64 = 60.0
const HOURS_IN_DAY float64 = 24.0
const DAYS_IN_WEEK float64 = 7.0

const CONFIG_FILE string = "config.txt"
const CONFIG_FROM_FILE bool = true

const LEARNING_RATE = "learningRate"
const NUM_HIDDEN_NODES = "numHiddenNodes"
const NUM_INPUT_NODES = "numInputNodes"
const NUM_OUTPUT_NODES = "numOutputNodes"
const NUM_TEST_CASES = "numTestCases"
const EXTERNAL_TEST_DATA = "externalTestData"
const TEST_DATA_FILE = "testDataFile"
const TRAIN_MODE = "trainMode"
const WEIGHT_INIT = "weightInit"
const WRITE_WEIGHTS = "writeWeights"
const FILE_NAME = "fileName"
const WEIGHT_LOWER_BOUND = "weightLowerBound"
const WEIGHT_UPPER_BOUND = "weightUpperBound"
const ERROR_THRESHOLD = "errorThreshold"
const MAX_ITERATIONS = "maxIterations"
const WEIGHT_SAVE_EVERY = "weightSaveEvery"
const KEEP_ALIVE_EVERY = "keepAliveEvery"

const CONFIG_PREFIX = ":"

type NetworkParameters struct
{
   learningRate float64
   
   numHiddenNodes int
   numInputNodes  int
   numOutputNodes int

   numTestCases     int
   externalTestData bool
   testDataFile     string
   
   trainMode  bool
   weightInit int // 1 is randomize, 2 is zero, 3 is manual, 4 is load from file
   writeWeights bool
   fileName string
   
   weightLowerBound float64
   weightUpperBound float64
   
   errorThreshold  float64
   maxIterations   int
   weightSaveEvery int
   keepAliveEvery  int
} // type NetworkParameters struct

type NetworkArrays struct
{
   a                        *[]float64
   h                        *[]float64
   F                        *[]float64
   inputHiddenWeights       *[][]float64
   hiddenOutputWeights      *[][]float64
   thetaJ                   *[]float64
   omegaJ                   *[]float64
   psiJ                     *[]float64
   thetaI                   *[]float64
   omegaI                   *[]float64
   psiI                     *[]float64
} // type NetworkArrays struct

var parameters NetworkParameters
var arrays NetworkArrays
var truthTable [][]float64
var expectedOutputs [][]float64

var trainStart time.Time
var done bool
var epochError float64
var inputError float64
var epoch int
var executionTime float64

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
   
   arrays, truthTable, expectedOutputs = allocateNetworkMemory()
   populateNetworkMemory()
   
   if (parameters.trainMode)
   {
      train(truthTable, expectedOutputs)
   }
   
   reportResults()

   if (parameters.writeWeights)
   {
      saveWeights()
   }
} // func main()

/**
 * The loadNetworkParameters function reads the network configuration parameters from a file and sets the global `parameters`
 * structure accordingly. The function reads the configuration file line by line, parsing each line to extract the parameter
 * name and value. It then sets the corresponding parameter in the `parameters` structure.
 *
 * The config file is assumed to have all necessary parameters set with a prefix + parameter name and a space-separated value.
 * Therefore, the function looks for the prefix + parameter name at the beginning of each line and sets the corresponding
 * parameter in the `parameters` structure.
 *
 * Process:
 * 1. The function checks if the configuration file exists. If not, the function panics.
 * 2. The function opens the configuration file in read-only mode and creates a scanner to read the file line by line.
 * 3. The function parses each line to extract the parameter name and value, then sets the corresponding parameter in
 *    the `parameters`
 *
 * Syntax:
 * - os.Stat(filename string) checks if the file exists.
 * - os.OpenFile(filename string, flag int, perm os.FileMode) opens a file for writing.
 * - error.Is(err error, target error) checks if the error is equal to the target error.
 * - defer file.Close() defers the file's closure until the function returns.
 * - bufio.NewScanner(file *os.File) creates a new scanner to read the file line by line.
 * - scanner.Scan() reads the next line from the file.
 * - scanner.Text() returns the current line from the scanner.
 * - strings.HasPrefix(s, prefix string) checks if the string starts with the given prefix.
 * - strings.Fields(s string) splits the string into fields separated by whitespace.
 *
 * Limitations:
 * - Assumes that the configuration file exists and is correctly formatted.
 * - Assumes that the configuration file contains all necessary parameters and that the values are correctly formatted.
 */
func loadNetworkParameters()
{
   var file *os.File
   var err error
   var fileExists bool = false
   var configLine string

   _, err = os.Stat(CONFIG_FILE)
   if (err == nil)
   {
      fileExists = true
   }

   if (!fileExists)
   {
      panic("Configuration file does not exist!")
   }

   file, err = os.OpenFile(CONFIG_FILE, os.O_RDONLY, 0644) // open file in read-only mode
   checkError(err)

   defer file.Close()

   var scanner *bufio.Scanner = bufio.NewScanner(file)

   parameters = NetworkParameters{}

   for (scanner.Scan())
   {
      configLine = scanner.Text()

      if (strings.HasPrefix(configLine, CONFIG_PREFIX))
      {
         parts := strings.Fields(configLine)
         if (len(parts) == 2)
         {
            variableName := parts[0]
            variableValue := parts[1]

            switch variableName
            {
               case CONFIG_PREFIX + LEARNING_RATE:
                  parameters.learningRate, _ = strconv.ParseFloat(variableValue, 64)
                  break

               case CONFIG_PREFIX + NUM_HIDDEN_NODES:
                  parameters.numHiddenNodes, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + NUM_INPUT_NODES:
                  parameters.numInputNodes, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + NUM_OUTPUT_NODES:
                  parameters.numOutputNodes, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + NUM_TEST_CASES:
                  parameters.numTestCases, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + EXTERNAL_TEST_DATA:
                  parameters.externalTestData, _ = strconv.ParseBool(variableValue)
                  break

               case CONFIG_PREFIX + TEST_DATA_FILE:
                  parameters.testDataFile = variableValue
                  break

               case CONFIG_PREFIX + TRAIN_MODE:
                  parameters.trainMode, _ = strconv.ParseBool(variableValue)
                  break

               case CONFIG_PREFIX + WEIGHT_INIT:
                  parameters.weightInit, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + WRITE_WEIGHTS:
                  parameters.writeWeights, _ = strconv.ParseBool(variableValue)
                  break

               case CONFIG_PREFIX + FILE_NAME:
                  parameters.fileName = variableValue
                  break

               case CONFIG_PREFIX + WEIGHT_LOWER_BOUND:
                  parameters.weightLowerBound, _ = strconv.ParseFloat(variableValue, 64)
                  break

               case CONFIG_PREFIX + WEIGHT_UPPER_BOUND:
                  parameters.weightUpperBound, _ = strconv.ParseFloat(variableValue, 64)
                  break

               case CONFIG_PREFIX + ERROR_THRESHOLD:
                  parameters.errorThreshold, _ = strconv.ParseFloat(variableValue, 64)
                  break

               case CONFIG_PREFIX + MAX_ITERATIONS:
                  parameters.maxIterations, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + WEIGHT_SAVE_EVERY:
                  parameters.weightSaveEvery, _ = strconv.Atoi(variableValue)
                  break

               case CONFIG_PREFIX + KEEP_ALIVE_EVERY:
                  parameters.keepAliveEvery, _ = strconv.Atoi(variableValue)
                  break
            } // switch variableName
         } // if (len(parts) == 2)
      } // if (strings.HasPrefix(configLine, CONFIG_PREFIX))
   } // for (scanner.Scan())
} // func loadNetworkParameters()

/**
 * The setNetworkParameters function initializes the network's configuration by defining its parameters within a
 * NetworkParameters structure. This setup includes learning rate, number of nodes in each layer
 * (input, hidden, output), number of test cases, training mode status, weight initialization method, bounds for weight values,
 * error threshold, and the maximum number of iterations.
 *
 * Parameters Defined:
 * - learningRate: Learning rate (lambda) for the gradient descent rate.
 * - numHiddenLayers, numInputNodes, numShallowHiddenNodes, numDeepHiddenNodes,
 * -     numOutputNodes: Specify the architecture of the network in terms of neuron counts.
 * - numTestCases: The number of test cases to be used in training/validation.
 * - externalTestData: Boolean indicating if the test data is external to the program.
 * - testDataFile: The name of the file containing the test data.
 * - trainMode: Boolean indicating if the network is in training mode.
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
      learningRate:     0.3,
      numInputNodes:    2,
      numHiddenNodes:   5,
      numOutputNodes:   3,
      numTestCases:     4,
      externalTestData: true,
      testDataFile:     "data.txt",
      trainMode:        true,
      weightInit:       1,
      writeWeights:     true,
      fileName:         "weights.txt",
      weightLowerBound: 0.1,
      weightUpperBound: 1.5,
      errorThreshold:   2e-4,
      maxIterations:    100000,
      weightSaveEvery:  10000,
      keepAliveEvery:   100000,
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
 * - Network architecture detailed by the count of input, hidden, and output nodes.
 * - The total number of test cases used for training or validation.
 * - Training mode indicator (true for training mode).
 * - Test data source indicator (true for external test data).
 * - Weight initialization method, where "1" denotes random initialization and "2" denotes initialization to zero, 3 denotes
 * -    manual initialization, and 4 denotes loading from a file.
 * - Range for random weight initialization, specifying lower and upper bounds.
 * - Keep alive and save weights every N iterations.
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

   fmt.Printf("Network: %d-%d-%d, NumberTestCases: %d\n",
      parameters.numInputNodes, parameters.numHiddenNodes, parameters.numOutputNodes, parameters.numTestCases)

   fmt.Printf("Train Mode: %t\n", parameters.trainMode)
   fmt.Printf("Weight Init: %d -- 1 = random, 2 = zero, 3 = manual, 4 = load from file\n", parameters.weightInit)
   fmt.Printf("Test Data: %t -- true = external, 2 = internal\n", parameters.externalTestData)
   fmt.Printf("Keep Alive Every: %d, Save Weights Every: %d\n", parameters.keepAliveEvery, parameters.weightSaveEvery)
   fmt.Printf("Random Range [%v, %v]\n\n", parameters.weightLowerBound, parameters.weightUpperBound)
} // func echoNetworkParameters()

/**
 * The allocateNetworkMemory function is responsible for initializing and allocating memory for various arrays and matrices used
 * by the network, including those for input to hidden layer weights, hidden to output layer weights, and other structures
 * for training such as omegas, thetas, and psis. The function also allocates the truth table for
 * inputs and expected outputs.

 * Returns:
 * - A NetworkArrays structure containing references to all allocated arrays and matrices used by the network.
 * - A truth table for network inputs as a slice of float64 slices.
 * - An output truth table as a slice of float64.

 * If the trainMode parameter is true, structures used exclusively for training (thetas, omegas, psis)
 * are allocated. This condition helps optimize memory usage by only allocating necessary arrays.

 * Limitations:
 * - Assumes that the global `parameters` structure is correctly initialized before this function is called.
 */
func allocateNetworkMemory() (NetworkArrays, [][]float64, [][]float64)
{
   var k, j, input, output int
   
   var a []float64 = make([]float64, parameters.numInputNodes)
   var inputHiddenWeights [][]float64 = make([][]float64, parameters.numInputNodes)
   for k = range inputHiddenWeights
   {
      inputHiddenWeights[k] = make([]float64, parameters.numHiddenNodes)
   }

   var h []float64 = make([]float64, parameters.numHiddenNodes)

   var hiddenOutputWeights [][]float64 = make([][]float64, parameters.numHiddenNodes)
   for j = range hiddenOutputWeights
   {
      hiddenOutputWeights[j] = make([]float64, parameters.numOutputNodes)
   }

   var F []float64 = make([]float64, parameters.numOutputNodes)
   
   var thetaJ, omegaJ, psiJ, thetaI, omegaI, psiI []float64
   
   if (parameters.trainMode)
   {
      thetaJ = make([]float64, parameters.numHiddenNodes)
      omegaJ = make([]float64, parameters.numHiddenNodes)
      psiJ = make([]float64, parameters.numHiddenNodes)
      thetaI = make([]float64, parameters.numOutputNodes)
      omegaI = make([]float64, parameters.numOutputNodes)
      psiI = make([]float64, parameters.numOutputNodes)
   } // if (parameters.trainMode)
   
   var inputTruthTable [][]float64 = make([][]float64, parameters.numTestCases)
   for input = range inputTruthTable
   {
      inputTruthTable[input] = make([]float64, parameters.numInputNodes)
   }
   
   var outputTruthTable [][]float64 = make([][]float64, parameters.numTestCases)
   for output = range outputTruthTable
   {
      outputTruthTable[output] = make([]float64, parameters.numOutputNodes)
   }
   
   return NetworkArrays
   {
      a:                              &a,
      h:                              &h,
      inputHiddenWeights:             &inputHiddenWeights,
      hiddenOutputWeights:            &hiddenOutputWeights,
      F:                              &F,
      thetaJ:                         &thetaJ,
      omegaJ:                         &omegaJ,
      psiJ:                           &psiJ,
      thetaI:                         &thetaI,
      omegaI:                         &omegaI,
      psiI:                           &psiI,
   }, inputTruthTable, outputTruthTable
} // func allocateNetworkMemory() NetworkArrays

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
   var test, k, i int

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
      parts := strings.Fields(testLine)
      if (len(parts) == parameters.numInputNodes + parameters.numOutputNodes + 1)
      {
         for k = 0; k < parameters.numInputNodes; k++
         {
            truthTable[test][k], _ = strconv.ParseFloat(parts[k], 64)
         }
         for i = 0; i < parameters.numOutputNodes; i++
         {
            expectedOutputs[test][i], _ = strconv.ParseFloat(parts[i + parameters.numInputNodes + 1], 64)
         }
      } // if (len(parts) == parameters.numInputNodes + parameters.numOutputNodes + 1)
      test++
   } // for (scanner.Scan() && test < parameters.numTestCases)
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
   var k, j, i int
   
   if (parameters.weightInit == 1)
   {
      rand.Seed(time.Now().UnixNano())
      for k = range *arrays.inputHiddenWeights
      {
         for j = range (*arrays.inputHiddenWeights)[k]
         {
            (*arrays.inputHiddenWeights)[k][j] = randomNumber(parameters.weightLowerBound, parameters.weightUpperBound)
         }
      } // for k = range *arrays.inputHiddenWeights

      for j = range *arrays.hiddenOutputWeights
      {
         for i = range (*arrays.hiddenOutputWeights)[j]
         {
            (*arrays.hiddenOutputWeights)[j][i] = randomNumber(parameters.weightLowerBound, parameters.weightUpperBound)
         }
      } // for j = range *arrays.hiddenOutputWeights
   } // if (parameters.weightInit == 1)

   if (parameters.weightInit == 3)
   {
      (*arrays.inputHiddenWeights)[0][0] = 0.8
      (*arrays.inputHiddenWeights)[0][1] = 0.5
      (*arrays.inputHiddenWeights)[1][0] = 0.5
      (*arrays.inputHiddenWeights)[1][1] = 0.5

      (*arrays.hiddenOutputWeights)[0][0] = -0.5
      (*arrays.hiddenOutputWeights)[1][0] = 0.5
   } // if (parameters.weightInit == 3)

   if (parameters.weightInit == 4)
   {
      loadWeights()
   } // if (parameters.weightInit == 4)
   
   if (parameters.externalTestData)
   {
      loadTestData()
   }
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
   }
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
func train(inputs [][]float64, expectedOutputs [][]float64)
{
   trainStart = time.Now()
   done = false
   
   epochError = math.MaxFloat64
   epoch = 0

   var input, j, k, i int
   var deltaWeightJI, deltaWeightKJ float64
   
   var a []float64 = *arrays.a
   var h []float64 = *arrays.h
   var F []float64 = *arrays.F

   var thetaJ []float64 = *arrays.thetaJ
   var omegaJ []float64 = *arrays.omegaJ
   var psiJ []float64 = *arrays.psiJ

   var thetaI []float64 = *arrays.thetaI
   var omegaI []float64 = *arrays.omegaI
   var psiI []float64 = *arrays.psiI

   var inputHiddenWeights [][]float64 = *arrays.inputHiddenWeights
   var hiddenOutputWeights [][]float64 = *arrays.hiddenOutputWeights
   
   for (!done)
   {
      epochError = 0.0
      for input = 0; input < parameters.numTestCases; input++
      {
         inputError = 0.0

         runTrain(&a, &h, &F, &thetaJ, &thetaI, &omegaI, &psiI, &inputHiddenWeights, &hiddenOutputWeights, &inputs[input],
                  &expectedOutputs[input])
         
         for i = 0; i < parameters.numOutputNodes; i++
         {
            inputError += omegaI[i] * omegaI[i]
         } // for i = 0; i < parameters.numOutputNodes; i++
         
         inputError /= 2.0
         
         for j = 0; j < parameters.numHiddenNodes; j++
         {
            omegaJ[j] = 0.0
            for i = 0; i < parameters.numOutputNodes; i++
            {
               omegaJ[j] += psiI[i] * hiddenOutputWeights[j][i]
               deltaWeightJI = parameters.learningRate * h[j] * psiI[i]
               hiddenOutputWeights[j][i] += deltaWeightJI
            } // for i = 0; i < parameters.numOutputNodes; i++

            psiJ[j] = omegaJ[j] * activationPrime(thetaJ[j])
            
            for k = 0; k < parameters.numInputNodes; k++
            {
               deltaWeightKJ = parameters.learningRate * a[k] * psiJ[j]
               inputHiddenWeights[k][j] += deltaWeightKJ
            } // for k = 0; k < parameters.numInputNodes; k++
         } // for j = 0; j < parameters.numHiddenNodes; j++
         
         runTrain(&a, &h, &F, &thetaJ, &thetaI, &omegaI, &psiI, &inputHiddenWeights, &hiddenOutputWeights, &inputs[input],
                  &expectedOutputs[input])
         
         inputError = 0.0

         for i = 0; i < parameters.numOutputNodes; i++
         {
            inputError += omegaI[i] * omegaI[i]
         } // for i = 0; i < parameters.numOutputNodes; i++

         inputError /= 2.0
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
      fmt.Printf("Input: %v, Expected: %f, Predicted: %f\n", input, expectedOutputs[index], run(input))
   }
} // func reportResults()

/**
 * The run function performs forward propagation through the network for a given input array `a`, computing the network's output.
 * The function uses the input array to compute dot products between the input neurons and the weights, eventually computing the
 * value of the output node (the network's output).
 *
 * Process Overview:
 * 1. Computes weighted sums at the hidden nodes, applying the sigmoid activation function to each sum/theta.
 * 2. Collects the hidden activations, again applying weights and the sigmoid function to produce the final output array.
 * 3. Returns the network's predictions for the given input.
 *
 * Parameters:
 * - `a`: Input array to use to make a prediction.
 *
 * Limitations:
 * - Assumes that the network's weights (`inputHiddenWeights` and `hiddenOutputWeights`) have been properly initialized.
 * - Assumes that the input array `a` matches the size expected by the network.
 */
func run(a []float64) []float64
{
   var j, k, i int
   
   var h []float64 = *arrays.h
   var F []float64 = *arrays.F
   var inputHiddenWeights [][]float64 = *arrays.inputHiddenWeights
   var hiddenOutputWeights [][]float64 = *arrays.hiddenOutputWeights
   
   for j = 0; j < parameters.numHiddenNodes; j++
   {
      var sum float64 = 0.0
      for k = 0; k < parameters.numInputNodes; k++
      {
         sum += a[k] * inputHiddenWeights[k][j]
      }
      h[j] = activationFunction(sum)
   } // for j = 0; j < parameters.numHiddenNodes; j++
   
   for i = 0; i < parameters.numOutputNodes; i++
   {
      var sum float64 = 0.0
      for j = 0; j < parameters.numHiddenNodes; j++
      {
         sum += h[j] * hiddenOutputWeights[j][i]
      }
      F[i] = activationFunction(sum)
   } // for i = 0; i < parameters.numOutputNodes; i++
   
   return *arrays.F
} // func run(a []float64) []float64

/**
 * The runTrain function is designed for the forward propagation part of the neural network training process.
 * It updates the network's hidden layer outputs and the final output array (F) value based on a given input array.
 *
 * Process:
 *    1. Copies the input vector into the network's input layer.
 *    2. For each hidden neuron, computes the weighted sum of its inputs and applies the sigmoid function to
 *       obtain the neuron's output.
 *    3. Calculates the weighted sum of the hidden layer outputs and applies the sigmoid function to determine the network outputs.
 *    4. Computes the error between the small omega i array and the psi i array used for network training.
 *
 * Parameters:
 * - `a`: Reference to the input layer vector.
 * - `h`: Reference to the hidden layer output vector.
 * - `F`: Reference to the final output vector.
 * - `thetaJ`: Reference to the variable storing the weighted sum of the hidden layer outputs before applying sigmoid.
 * - `thetaI`: Reference to the variable storing the weighted sum of the output layer outputs before applying sigmoid.
 * - `omegaI`: Reference to the vector storing the error between expected and predicted outputs.
 * - `psiI`: Reference to the vector storing the error gradient for the output layer.
 * - `inputHiddenWeights`: Reference to the matrix of weights from the input layer to the hidden layer.
 * - `hiddenOutputWeights`: Reference to the vector of weights from the hidden layer to the output neuron.
 * - `input`: The input vector for the current training example.
 * - `outputs`: The expected outputs for the current training example.
 *
 * Limitations and Conditions:
 * - Assumes that the network's weights (`inputHiddenWeights` and `hiddenOutputWeights`) have been properly initialized.
 */
func runTrain(a *[]float64, h *[]float64, F *[]float64, thetaJ *[]float64, thetaI *[]float64, omegaI *[]float64, psiI *[]float64,
              inputHiddenWeights *[][]float64, hiddenOutputWeights *[][]float64, input *[]float64, outputs *[]float64)
{
   var j, k, i int
   
   for k = 0; k < parameters.numInputNodes; k++
   {
      (*a)[k] = (*input)[k]
   } // for k = 0; k < parameters.numInputNodes; k++
   
   for j = 0; j < parameters.numHiddenNodes; j++
   {
      (*thetaJ)[j] = 0.0
      for k = 0; k < parameters.numInputNodes; k++
      {
         (*thetaJ)[j] += (*a)[k] * (*inputHiddenWeights)[k][j]
      } // for k = 0; k < parameters.numInputNodes; k++
      
      (*h)[j] = activationFunction((*thetaJ)[j])
   } // for j = 0; j < parameters.numHiddenNodes; j++
   
   for i = 0; i < parameters.numOutputNodes; i++
   {
      (*thetaI)[i] = 0.0
      for j = 0; j < parameters.numHiddenNodes; j++
      {
         (*thetaI)[i] += (*h)[j] * (*hiddenOutputWeights)[j][i]
      } // for j = 0; j < parameters.numHiddenNodes; j++

      (*F)[i] = activationFunction((*thetaI)[i])
      (*omegaI)[i] = (*outputs)[i] - (*F)[i]
      (*psiI)[i] = (*omegaI)[i] * activationPrime((*thetaI)[i])
   } // for i = 0; i < parameters.numOutputNodes; i++
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
}

/**
 * The saveWeights function writes the network's weights to a file. The function opens a file for writing and writes the
 * network's architecture and weights to the file. The architecture is written as a string in the format "input-hidden-output".
 * The input-hidden weights are written first, followed by the hidden-output weights.
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
 * 4. Writes the input-hidden weights to the file.
 * 5. Writes the hidden-output weights to the file.
 * 6. Closes the file.
 */
func saveWeights()
{
   var j, k, i int
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

   _, err = file.WriteString(fmt.Sprintf("%d-%d-%d\n", parameters.numInputNodes, parameters.numHiddenNodes,
                             parameters.numOutputNodes)) // write network architecture to file
   checkError(err)

   _, err = file.WriteString("\n") // write new line to file
   checkError(err)

   for k = 0; k < parameters.numInputNodes; k++
   {
      for j = 0; j < parameters.numHiddenNodes; j++
      {
         _, err = file.WriteString(fmt.Sprintf("%f\n", (*arrays.inputHiddenWeights)[k][j]))
         checkError(err)
      } // for j = 0; j < parameters.numHiddenNodes; j++
   } // for k = 0; k < parameters.numInputNodes; k++

   _, err = file.WriteString("\n") // write new line to file
   checkError(err)

   for j = 0; j < parameters.numHiddenNodes; j++
   {
      for i = 0; i < parameters.numOutputNodes; i++
      {
         _, err = file.WriteString(fmt.Sprintf("%f\n", (*arrays.hiddenOutputWeights)[j][i]))
         checkError(err)
      } // for i = 0; i < parameters.numOutputNodes; i++
   } // for j = 0; j < parameters.numHiddenNodes; j++
}

/**
 * The loadWeights function reads the network's weights from a file. The function opens a file for reading and reads the
 * network's architecture and weights from the file. The architecture is read as a string in the format "input-hidden-output".
 * The input-hidden weights are read first, followed by the hidden-output weights. The function then checks if the network's
 * architecture matches the architecture in the weights file before reading in the weights.
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
   var j, k, i int
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

   var numberInputNodes string = strconv.Itoa(parameters.numInputNodes)
   var numberHiddenNodes string = strconv.Itoa(parameters.numHiddenNodes)
   var numberOutputNodes string = strconv.Itoa(parameters.numOutputNodes)

   var configString string = numberInputNodes + "-" + numberHiddenNodes + "-" + numberOutputNodes

   if (configString != architecture)
   {
      fmt.Println("Network architecture does not match the architecture in the weights file!")
      panic(err)
   } // if (configString != architecture)

   _, err = fmt.Fscan(file, architecture)

   for k = 0; k < parameters.numInputNodes; k++
   {
      for j = 0; j < parameters.numHiddenNodes; j++
      {
         _, err = fmt.Fscan(file, &(*arrays.inputHiddenWeights)[k][j])
         checkError(err)
      } // for j = 0; j < parameters.numHiddenNodes; j++
   } // for k = 0; k < parameters.numInputNodes; k++

   _, err = fmt.Fscan(file, architecture)

   for j = 0; j < parameters.numHiddenNodes; j++
   {
      for i = 0; i < parameters.numOutputNodes; i++
      {
         _, err = fmt.Fscan(file, &(*arrays.hiddenOutputWeights)[j][i])
         checkError(err)
      } // for i = 0; i < parameters.numOutputNodes; i++
   } // for j = 0; j < parameters.numHiddenNodes; j++
}