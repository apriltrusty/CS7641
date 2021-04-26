Code is saved at: https://github.com/apriltrusty/CS7641/tree/master/Assignment%204

To run, use:
        python3 car_rental.py
        python3 frozen_lake.py

These write to a directory called Excel/ which will need to be already present.

My PI and VI code are modified versions of these algorithms from Brett Daley's Gym Classics library: https://github.com/brett-daley/gym-classics
I modified dynamic_programming.py from the above to support the Frozen Lake environment from OpenAI Gym and output additional information.

My Q-Learning code is based on https://github.com/brett-daley/gym-classics/tree/0458841a84c1fb9f47d3dd3d8dfb217b5c189e3e#example-reinforcement-learning

To run, you will need the following libraries:
        1. OpenAI Gym
        2. Gym Classics
        3. NumPy
        4. Pandas
        5. Openpyxl (for writing to Excel .xlsx files)