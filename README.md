# Mislocalization in plasmon-enhanced single-molecule imaging, and other stuff...

This package is the workhorse for my PhD, which means it is mostly developed as a standalone python package to model plasmon-enhanced single-molecule imaging experiments, but it does other random stuff too. First of which is make some figures I'm proud of, like the ones in [my publication with Curly](http://pubs.acs.org/articlesonrequest/AOR-d4zR6Ppm6k3QbIRtJdNk) from Julie Biteen's group at the University of Michigan. 

Some people have downloaded this and got pieces to work for them, which is good! but there are a number of dependencies that are only required for specific functionalities, so an all encompassing installation doesn't make much sense. Really if that's the case then this should probably be spit into multiple repositories/pieces of software, but I guess that's a lesson learned. Currently working on getting all paths that will need to be modified upon installation to the package `__init__.py`, but right now it's just the path to the MNPBEM17 package for matlab for running simulations. 

Examples for use can be found in the seperate repo of Jupyter notebooks, `mispolarization`. Send me a messege if you are interested! Happy to help. 