# Contribution Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, refer to [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and learn about the contribution process of the source code repository.

Developers should focus on the following points when preparing local code and submitting PRs:

1. When submitting a PR, carefully fill in the business background, purpose, solution, and other information according to the PR template.
2. If your modification is not a simple bug fix but involves adding new features, new interfaces, new configuration parameters, or modifying code flows, initiate a solution discussion through an Issue first to avoid your code being rejected. If you are unsure whether your modification can be classified as a "simple bug fix," you can also submit an Issue for solution discussion.

Developer contribution scenarios mainly include:

## 1. Contributing New Operators

If you have a new operator that you want to design and implement based on NPU, you are welcome to propose your ideas and design solutions in an Issue. The complete contribution process is as follows:

### 1. Create an Issue Requirement

Create a `Requirement|Requirement Suggestion` type Issue and explain the design solution of the new operator. The Issue generally needs to include the following content:

- **Background Information**
- **Value/Role**
- **Design Solution**

Comment `/assign @yourself` in the submitted Issue to claim the task.

### 2. Requirement Review

The SIG group will assign a Committer to review your submitted Issue and provide feedback. After completing the modifications, @ the corresponding Committer in the Issue.

If the requirement is accepted, [SIG members](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md) will assign an appropriate operator classification path (for example, `experimental/image`). Submit the contributed operator to the corresponding operator classification directory under `experimental`.

### 3. PR Submission

The minimal ecosystem operator deliverables are as follows:

```Cpp
${op_class}                                          # Operator classification
├── ${op_name}                                       # Operator name
│   ├── ${op_name}.cpp                               # Operator Kernel implementation file
│   └── tests                                        
│   │   ├── test_${op_name}.py                       # Operator test file
│   ├── CMakeLists.txt                               # Operator compilation configuration file
│   ├── README.md                                    # Operator README documentation
```

PR submission requirements:

- Code deliverables: Operator Kernel implementation and operator test files must be provided. For the development process, refer to [fast_kernel_launch_example](https://gitcode.com/cann/ops-nn/blob/master/examples/fast_kernel_launch_example/README.md).
- Documentation deliverables: Operator README documentation is mandatory. Other documents can be provided as needed. For documentation writing templates and specifications, refer to the [Documentation Contribution Guide](https://gitcode.com/cann/ops-nn/blob/master/docs/CONTRIBUTING_DOCS.md).
- Precision requirements: Newly contributed operators must meet precision standards. For details, refer to the [Ecosystem Operator Open-Source Precision Standard](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md).
- Compliance check:
  - Whether the code complies with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
  - Whether the code compiles successfully
  - Whether the Markdown document syntax complies with specifications
- Contribution directory: Submit to the specified directory `experimental/${op_class}` according to SIG members' opinions. You can refer to the file placement rules of existing operators.
- PR submission: Submit a PR to the target branch through `git` commands. Check whether the PR title is clear, whether the PR description is standardized (specifying the changed content and reasons, whether it is associated with the corresponding Issue), and whether the CLA is signed.

If you want to contribute project standard operators, the deliverables and development process are more complex than ecosystem operators, including Kernel and Tiling implementation. For detailed contribution guidance, refer to the [Appendix](#appendix).

### 4. CI Gate

Trigger the open-source repository gate by commenting with the `compile` command, and make modifications based on the CI detection results. Currently, the CI gate includes the following checks:

- Code compilation
- Static check (if codecheck false positives are involved, submit them to SIG members for shielding)
- UT test
- Smoke test

After the gate passes, @ the assigned Committer in the associated Issue.

### 5. Committer Review

After the Committer reviews, feedback will be provided. Modify according to the feedback, and @ the assigned Committer after completion.

### 6. Maintainer Merge

After the Committer review passes, the `/lgtm` label is marked. The Maintainer will conduct the final review within 1 day. After confirming no issues, the `/approve` label is marked to merge the PR.

## 2. Operator Bug Fix

If you find certain operator bugs in this project and want to fix them, you are welcome to create a new Issue for feedback and tracking.

You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#submit-issuehandle-issue-task) guide to create a `Bug-Report|Bug Feedback` type Issue to describe the bug, and then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for processing.

## 3. Operator Optimization

If you have generalization enhancement or performance optimization ideas for the implementation of certain operators in this project and want to implement these optimization points, you are welcome to contribute to operator optimization.

You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#submit-issuehandle-issue-task) guide to create a `Requirement|Requirement Suggestion` type Issue to explain the optimization points and provide your design solution, and then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for tracking optimization.

## 4. Documentation Error Correction

If you find description errors in certain operator documentation in this project, you are welcome to create a new Issue for feedback and correction. For documentation specifications, refer to the [Documentation Contribution Guide](https://gitcode.com/cann/ops-nn/blob/master/docs/CONTRIBUTING_DOCS.md).

You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#submit-issuehandle-issue-task) guide to create a `Documentation|Documentation Feedback` type Issue to point out the problems in the corresponding documentation, and then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself to correct the corresponding documentation description.

## 5. Help Resolve Others' Issues

If you have an appropriate solution to a problem encountered by others in the community, you are welcome to post a comment in the Issue to communicate, help others solve problems and pain points, and jointly optimize usability.

If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself and track and assist in solving the problem.

## Appendix

The project standard operator deliverables are as follows:

```Cpp
${op_class}                                          # Operator classification
├── ${op_name}                                       # Operator name
│   ├── op_host                                      # Operator definition and Tiling-related implementation
│   │   ├── ${op_name}_def.cpp                       # Operator definition file
│   │   ├── ${op_name}_tiling.cpp                    # Operator Tiling implementation file
│   │   └── CMakeLists.txt
│   ├── op_kernel                                    # Operator Kernel directory
│   │   ├── ${op_name}.cpp                           # Kernel entry file, containing the main function and scheduling logic
│   │   ├── ${op_name}.h                             # Kernel implementation file, defining the Kernel header file, containing function descriptions, structure definitions, and logic implementation
│   │   ├── ${op_name}_tiling_data.h                 # TilingData file, storing Tiling strategy-related configuration information
│   │   └── ${op_name}_tiling_key.h                  # TilingKey file, defining the key of the Tiling strategy to identify different partitioning methods
│   ├── CMakeLists.txt                               # Operator compilation configuration file, keep the original file
│   └── README.md                                    # Operator description document
│   └── tests                                        # Operator test files
│   │   ├── ut                                       # Operator UT test files
```

PR submission requirements:

- Code deliverables: op_host operator Tiling implementation, op_kernel operator Kernel implementation, and operator UT test files must be provided. For the development process, refer to the [Operator Development Guide](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/develop/aicore_develop_guide.md).
- Documentation deliverables: Operator README documentation is mandatory. Other documents can be provided as needed. For documentation writing templates and specifications, refer to the [Documentation Contribution Guide](https://gitcode.com/cann/ops-nn/blob/master/docs/CONTRIBUTING_DOCS.md).
- Compliance check:
  - Whether the code complies with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md) and the standard operator basic programming specifications
  - Whether the code compiles successfully
  - Whether the Markdown document syntax complies with specifications
- Contribution directory: Submit to the specified directory `experimental/${op_class}` according to SIG members' opinions. You can refer to the file placement rules of existing operators.
- PR submission: Submit a PR to the target branch through `git` commands. Check whether the PR title is clear, whether the PR description is standardized (specifying the changed content and reasons, whether it is associated with the corresponding Issue), and whether the CLA is signed."
