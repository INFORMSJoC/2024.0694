[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)


# [Approximate Resolution of Stochastic Choice-based Discrete Planning](https://doi.org/10.1287/ijoc.2024.0694)

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

This repository contains supporting material for the paper 
 
    "Approximate Resolution of Stochastic Choice-based Discrete Planning" by Jiajie Zhang, Yun Hui Lin, and Gerardo Berbeglia.

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper.


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0694

https://doi.org/10.1287/ijoc.2024.0694.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{Lin2024,
  author =        {J. Zhang, Y.H. Lin, and G. Berbeglia},
  publisher =     {INFORMS Journal on Computing},
  title =         {Approximate Resolution of Stochastic Choice-based Discrete Planning},
  year =          {2025},
  doi =           {10.1287/ijoc.2024.0694},
  url =           {https://github.com/INFORMSJoC/2024.0694},
}
```


## Description
Stochastic choice-based discrete planning is a broad class of decision-making problems characterized by a sequential decision-making process involving a planner and a group of customers. The firm or planner first decides a subset of options to offer to the customers, who, in turn, make selections based on their utilities of those options. This problem has extensive applications in many areas, including assortment planning, product line design, and facility location. A key feature of these problems is that the firm cannot fully observe the customers’ utilities or preferences, which results from intrinsic and idiosyncratic uncertainties. Most works in the literature have studied a specific type of uncertainty, resulting in customized decision models that are subsequently tackled using ad-hoc algorithms designed to exploit the specific model structure.
In this paper we propose a modeling framework capable of solving this family of sequential problems that works for a large variety of uncertainties. We then leverage an approximation scheme and develop an adaptable mixed-integer linear programming method. To speed up the solution process, we further develop an efficient decomposition approach. We show that our solution framework can yield solutions proven to be (near-)optimal for a broad class of problems. We illustrate this by applying our approach to three classical application problems: constrained assortment optimization and two facility location problems. Through extensive computational experiments, we demonstrate the performance of our approach in terms of both solution quality and computational speed, and provide computational insights. In particular, when we use our method to solve the constrained assortment optimization problem under the Exponomial choice model, it improves the state-of-the-art.

This repository provides data for the problem and code for the proposed sampling-based Benders decomposition (**SBBD**) algorthm. The main folders are 'data', 'src', and 'results'.

- 'data': four datasets generator (and instances) used in the paper.

- 'src': the source code for the decomposition algorithm.

- 'results': high-resoultion figures in our paper

All experimental results can be found in the paper (manuscript and electronic companion).


## Replicating

- To run the code, you will need to make sure that you have already installed **Anaconda3**.

- You also need to install the solver **Gurobi 10.0.3** (license required).

Once the environment has set up, run **xxx_data.py** (i.e., CAOP_data.py, FLoP_data.py, MSMFLP_data.py) to generate the problem instance, and run **xxx_SBBD.py** for generating and solving the instance with the SBBD algorthm. 

For example, to generate the data for the CAOP instances, run 'CAOP_data.py'. To generate and solve CAOP instances, simply run 'CAOP_SBBD.py'.

The other tests also follow the same procedure. 


## Results

All results have been reported in the paper Sections 5-6 and Online Appendix. In addition, we report the figures of our paper in the Folder 'results'


## License

This software is released under the MIT license, which we report in file 'LICENSE'.
