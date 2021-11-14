# Kohonen-SOM-Algorithm
Kohonen  (SOM) Algorithm
By Shaked Gofin
Part A: Implementing the Kohonen algorithm and use it to fit a line of neurons to a data in the shape of a square.
In order of achieving the data points of a square with parameters of {(x,y) |  0 <= x <= 1, 0<=y<=1} I used np.random.rand(5000, 2) which returns 5000 pairs of randomize numbers between [0,1) thus giving us data points of a square as can be seen in the main function:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/51sLtgq/image.png" alt="image" border="0"></a>







I created a Class called SOM in order to incapsulate all the necessary functions.
It holds the Shape of the SOM, creates a line of neurons along the y=0 line and initiates the learning rate, time and sigma parameters. 

 <a href="https://imgbb.com/"><img src="https://i.ibb.co/ZgzJg0v/image.png" alt="image" border="0"></a>


The initial line of neurons can be seen here when I ran the program with zero iterations.

 <a href="https://imgbb.com/"><img src="https://i.ibb.co/L0MS83v/image.png" alt="image" border="0"></a>
 
The Train function sets the parameters and chooses a data point depending of if we asked for a Uniform distribution ( 1-Uniform , 0-NonUniform) 
Uniform - A random number to be as index for the given Data (in our case square or donut) 
NonUniform – the numbers are given a probability of being chosen as indexes for the data according to Dirichlet distribution.
 
 <a href="https://imgbb.com/"><img src="https://i.ibb.co/MBpDDfS/image.png" alt="image" border="0"></a>
 
The data point that we chose is sent to the find_bmu function as input_vector <x,y>.
The function goes over all neurons and calculates the Euclidean distance between the input vector and the neurons and adds the result along with the coordinates of the neuron that we checked to a list.
At the end we sort the results and return the neuron whose weight vector is most similar to the input vector also called the best matching unit (BMU).
We then can update the SOM using the update_som function.
 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/vh7gHq9/image.png" alt="image" border="0"></a>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/TbwHKVR/image.png" alt="image" border="0"></a> 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/0hX061c/image.png" alt="image" border="0"></a>
 

Here is a description of the Algorithm from the Wikipedia page:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/xMPmHfD/image.png" alt="image" border="0"></a> 






The results:
  

<a href="https://imgbb.com/"><img src="https://i.ibb.co/mHjpRSx/image.png" alt="image" border="0"></a> 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/0hNNVPZ/image.png" alt="image" border="0"></a>
  




Non-Uniform:
  

  
<a href="https://imgbb.com/"><img src="https://i.ibb.co/bJxcZ7L/image.png" alt="image" border="0"></a>  










Part A.2: Data from {<x,y> | 1<= x^2 +y^2 <= 2}
For the donut shape sampling I made the following function that return a desired amount of data points:

 <a href="https://imgbb.com/"><img src="https://i.ibb.co/BKcydnn/image.png" alt="image" border="0"></a>

And on the Main function:

 <a href="https://imgbb.com/"><img src="https://i.ibb.co/3BZvFK9/image.png" alt="image" border="0"></a>
 
Which produce the following results using 30 neurons:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Bgjh2nZ/image.png" alt="image" border="0"></a>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/7g8RD2K/image.png" alt="image" border="0"></a>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/vLYVgJk/image.png" alt="image" border="0"></a> 



Now if we use a grid of neurons:
15x15
 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/ZGFW3YR/image.png" alt="image" border="0"></a>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/VSyN8Vz/image.png" alt="image" border="0"></a>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/C05fFGT/iterations-550-neurons-30-shape-donut-uniform-1.png" alt="iterations-550-neurons-30-shape-donut-uniform-1" border="0" /></a>
 
