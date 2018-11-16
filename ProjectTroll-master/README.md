# ProjectTroll

* Train model:
  ```bash
  cj parrun train.py sherlock2 -dep . -alloc '--gres gpu:1 -p owners' -m 'train networks'
  ```

* Combine CSVs:
  ```bash
  cj reduce -f results/training_results.csv PID --header=1
  ```

* Get partial results to local machine:
  ```bash
  cj get PID/results
  ```

* Get the reproducible package to local machine:
  ```bash
  cj get PID
  cj save PID PATH
  ```
