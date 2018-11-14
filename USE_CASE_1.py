import deepeden as de

#################################################################################
#  USE CASE 1:                                                                  #
#                                                                               #
# Every machine learning problem seems to have five steps where a human         #
# actually makes a decision.                                                    #
#                                                                               #
# 1.  Specify a (modeling) problem.                                             #
# 2.  Specify a technique to solve the problem.                                 #
# 3.  Specify what information you wish to record during trainingself.          #
# 4.  Train.                                                                    #
# 5.  Summarize the results of training.                                        #
#                                                                               #
# Therefore I want my most basic machine learning exhibitions to take 5 lines   #
# of code.  The following is what the simplest program should look like.        #
#################################################################################

# Represents a pre-defined machine learning problem, the X-OR problem.  This
# problem requires a learning algorithm to learn the X-OR mapping
# (0,0) -> 0
# (0,1) -> 1
# (1,0) -> 1wf
# (1,1) -> 0
x_or_problem = de.problem.xor

# Loads a pre-defined machine learning technique called NEAT (Neuro-evolution
# of Augmenting Topologies) with certain specified parameters.
neat_technique = de.techniques.neat("species:5, mutation_rate:0.3, self_adjusting_learning:all")


# Defines a recorder for the training
my_recorder = de.recorders.basic_neat_recorder("specimens:top 3 every 5 generations, accuracy:every generation, loss:every generation")


# Runs the ML technique on the specified problem, returning a "history" of the
# training, which has data on accuracy/loss, and, in the case of NEAT, detailed
# information on all the neural networks which were generated during training.
neat_history = de.train(problem=x_or_problem, technique=neat_technique, recorder=my_recorder, time_allowed=60)


# summarize_all() is a convenience function allowing a quick-and-dirty summarization
# of what happened during the training--everything that was recorded by the
# recorder.
summary = neat_history.summarize_all()
