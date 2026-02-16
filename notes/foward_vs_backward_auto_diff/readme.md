# overview 

this stackoverflow explains why it's more cost saving to use backward auto diff compared to forward auto diff in case when n_output in function << n_input in function. 

[stackoverflow link](https://math.stackexchange.com/questions/2195377/reverse-mode-differentiation-vs-forward-mode-differentiation-where-are-the-be)

## time complexity
The key insight is 
    * while cost of evaluating all intermediate derivaties are the same 
    * The order of perform matrix multiplication on those intermediate derivaties make a difference
        * when computing matrix multiplication ABC, whether compute AB or BC fast has number of op impact depending on dimension of A, B,C
    * in deep learning, we expects to have dimension of parameter goes down as layer goes deeper. (finally reduce to loss dim= 1)
        * in this case, heuristically, backward AD will save cost in matrix multplication. 


### wiki's confusion:
* wiki says something like doing n_input sweep vs n_output sweep is the time difference, this is misleading. 

what wiki says: 
* wiki assumes in forward mode, you need to compute intermediate output as well in each sweep to maintain O(1) memory
    * then there is overlapping usage of intermediate output when computing derivaties with respect to different input 
    * And this is costly obviously if you need to compute intermediate output again 
* backward mode stores all intermediate output. thus has no issue with this. 
* **but** one can argue with cache intermediate output, cost of computing intermediate derivative will be the same. 
  
stackoverflow argues another time cost: 

* even if you cache all intermediate output -> thus same cost for computing intermediate derivatives
* There is still scope for optimizating number of ops in chain multplication of (derivative) matrices. 

## memory complexity
There is also memory consideration
* forward AD stores intermediate output on the fly, only current layer in/output and derivates needed
    * This is the worst variant, you compute same intermediate output multiple times. but save memory
* backward need to store all intermediate output/ derivatives. 
    * this does not save memory but save compute time.
    * plus additional benefit from matrix multiplication order


