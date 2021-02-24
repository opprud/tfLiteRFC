# RFC - SELU activation function for TF Lite Micro

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Morten Opprud (opprud@gmail.com)                     |
| **Sponsor**   | A N Expert (whomever@tensorflow.org)                 |
| **Updated**   | 2021-02-25                                           |

## Objective

Using SELU according to [Klambauer et al](https://arxiv.org/abs/1706.02515) in the TF Lite micro branch may provide a shortcut to BatchNorm like performance at less computational cost and complexity.

## Motivation
Since BatchNorm is not supported by TF lite micro, it seems using SeLu activation with Le_cun kernel initializer and normalisation of the input has a similar normalising effect, as suggested by Klaumbauer et.al.

Implementing SELU in the TF Lite micro branch may be easier than implementaiton of Batch Normalisation. 

Since SELU is just an activation function, less compute is required to achieve self normalising features in a CNN, than by using BN layers. 

This has the benfits of allowing implementation of deeper convolutional networks on resource constrained microcontrollers. Deeper CNN's often seems to utilize BN to normalise layers and reduce internal covariate shift in layers,   

CNN's applied on raw sensor data have in some cases proved superior to networks with comprehensive feature extraction, for detecting transcient events, e.g. https://www.researchgate.net/publication/314247372_A_New_Deep_Learning_Model_for_Fault_Diagnosis_with_Good_Anti-Noise_and_Domain_Adaptation_Ability_on_Raw_Vibration_Signals

## User Benefit

CNN's or other deep learning networks implemented in deep learning frameworks like Keras, where SELU is supported can be directly translated to TF Lite micro models

Self normalising of layers in deep models can be acheived.

## Design Proposal

It seems like the SELU can be implemented based upon one of the oterh availible activations availible in TF Lite micro.
The ELU activation works similarily, the SELU is just a scled version (hence the 'S' in SELU)

The ELU is implemented as a lookup table for `int8`, the SELU could be similar 

A `float` prototype is shown below
```
int selu(float *p, float *res) {
	float v;
	//iterate through array
	while (*p) {
		v = *p;
		if(v > 0.0)
		{
			*res = LAMBDA * v;
		}
		else
		{
			*res = LAMBDA * (ALPHA * exp(v) - ALPHA);
		}
		//increment pointers
		p++;
		res++;
	}
	return 0;
}
```

### Alternatives Considered
* Make sure to discuss the relative merits of alternatives to your proposal.

### Performance Implications
* Do you expect any (speed / memory)? How will you confirm?
* There should be microbenchmarks. Are there?
* There should be end-to-end tests and benchmarks. If there are not (since this is still a design), how will you track that these will be created?

### Dependencies
* Dependencies: No
* Dependent projects: No (i don't think)

### Engineering Impact
* It is expected that the size and required compute will be less than with an implementation of Batch Normalisation 
* A test similar to existing activation function should be written

### Platforms and Environments
* Platforms: does this work on all platforms supported by TensorFlow? Currently TF Lite Micro is unsupported, this aim of the RFC is to include the feature in this branch

### Best Practices
* Does this proposal change best practices for some aspect of using/developing TensorFlow? NO

### Tutorials and Examples
* If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
    - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
    - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer. 
    - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged. 

### Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

### User Impact
* What are the user-facing changes? How will this feature be rolled out?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
