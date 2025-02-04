param(
    [Parameter(Mandatory=$true)]
    [int]$registration_number
)

python task-01-simple-perceptron.py --registration_number $registration_number