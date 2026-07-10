# Datasheet for Multi-Agent Collusion and Correlated Failure Benchmark Dataset

## Motivation

### For what purpose was the dataset created?
This dataset was created to support research on multi-agent collusion detection and correlated failure analysis. It provides labeled traces from simulated multi-agent scenarios to train and evaluate detectors for identifying collusive behavior.

### Who created the dataset and on behalf of whom?
Swapnanil Adhikary, as part of research for the NeurIPS 2027 Evaluations & Datasets Track.

## Composition

### What do the instances that comprise the dataset represent?
Each instance represents a single action taken by an agent in a multi-agent scenario. Instances include the agent's action, observation, reward, and a label indicating whether the agent was colluding.

### How many instances are there?
The dataset contains approximately 10,000 trace records across multiple scenarios, episodes, and seeds.

### Does the dataset contain all possible instances?
No, the dataset represents a sample of possible multi-agent interactions. It does not cover all possible collusion strategies or failure modes.

### What data does each instance consist of?
- `trace_id`: Unique identifier
- `scenario_type`: Type of scenario (bertrand, oversight)
- `episode`: Episode number
- `step`: Step number
- `agent_id`: Agent identifier
- `action_type`: Type of action
- `action_content`: Action details (JSON)
- `observation_content`: Observation details (JSON)
- `reward`: Reward received
- `done`: Episode completion flag
- `label`: colluding/non_colluding
- `label_reason`: Reason for label

### Is there a label or target associated with each instance?
Yes, each instance has a label indicating whether the agent was colluding (`colluding`) or not (`non_colluding`). Labels are determined based on behavioral metrics (collusion index, collusion rate).

### Are there recommended data splits?
We recommend splitting by episode: 70% training, 15% validation, 15% test. Ensure no episode appears in multiple splits.

## Collection Process

### How was the data collected?
Data was generated using the multicollude benchmark environment. Agents (dummy agents for this version) interact in simulated scenarios with controlled incentives and communication settings.

### What preprocessing was done?
None. Raw traces from the environment are included as-is.

### Was the data validated?
Yes, basic validation checks were performed:
- All required fields are present
- Labels are consistent with behavioral metrics
- No missing values in critical fields

## Uses

### What tasks can the dataset be used for?
1. Training collusion detectors
2. Evaluating detection algorithms
3. Studying multi-agent behavior patterns
4. Benchmarking new detection methods

### Is there anything about the composition of the dataset that might impact its use?
Yes:
- Synthetic data may not perfectly represent real-world collusion
- Limited to pricing and review scenarios
- Labels are behavioral, not based on intent

## Distribution

### Will the dataset be distributed to the public?
Yes, the dataset will be hosted on HuggingFace under CC-BY 4.0 license.

### How can the dataset be accessed?
- HuggingFace: https://huggingface.co/datasets/yourusername/multicollude
- GitHub: https://github.com/yourusername/multicollude

### Has the dataset been used for any tasks already?
Yes, in the accompanying paper for NeurIPS 2027.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?
Swapnanil Adhikary

### How can the dataset be updated?
Future versions will be released on HuggingFace with version numbering.

### If the dataset relates to people, are there applicable limits on re-identification?
Not applicable. The dataset contains no personal information.
