# ODE-Solver

ODE-solver using the method of multiple shooting

---


Table of contents
=================

<!--ts-->
   * [Prerequisites](#prerequisites)
   * [Short explanation](#short-explanation)
   * [Examples](#examples)
      * [Simple ODE](#stdin)
      * [Lotka-Volterra](#Lotka-Volterra)
   * [Ideas for improvement](#ideas-for-improvement)
<!--te-->

Prerequisites
=====

The code has been tested with Python 2.7 and Python 3.
Libraries used:
* numpy
* matplotlib (if you want to run the examples provided)

Short explanation
=====

Examples
=====

```python
f = open("daten.txt", "r")
data = f.read().split("\n")
for age in range(0, 122):
	print data[age*4+2], data[age*4+3]
```
Ideas for improvement
=====