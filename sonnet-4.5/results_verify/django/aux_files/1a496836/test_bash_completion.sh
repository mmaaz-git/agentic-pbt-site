#!/bin/bash

# Test script to understand how COMP_CWORD works in bash completion

echo "Testing COMP_CWORD behavior in bash completion"
echo "=============================================="

# Simulating different completion scenarios
echo -e "\nScenario 1: User types 'mycommand' and hits TAB"
echo "Command line: mycommand▌"
COMP_WORDS=("mycommand")
COMP_CWORD=1
echo "COMP_WORDS=(${COMP_WORDS[@]})"
echo "COMP_CWORD=$COMP_CWORD"
echo "Number of words: ${#COMP_WORDS[@]}"
echo "Accessing COMP_WORDS[COMP_CWORD-1]: ${COMP_WORDS[$((COMP_CWORD-1))]}"
echo ""

echo -e "\nScenario 2: User types 'mycommand arg1' and hits TAB"
echo "Command line: mycommand arg1▌"
COMP_WORDS=("mycommand" "arg1")
COMP_CWORD=2
echo "COMP_WORDS=(${COMP_WORDS[@]})"
echo "COMP_CWORD=$COMP_CWORD"
echo "Number of words: ${#COMP_WORDS[@]}"
echo "Accessing COMP_WORDS[COMP_CWORD-1]: ${COMP_WORDS[$((COMP_CWORD-1))]}"
echo ""

echo -e "\nScenario 3: User types 'mycommand ' (with space) and hits TAB"
echo "Command line: mycommand ▌"
COMP_WORDS=("mycommand" "")
COMP_CWORD=1
echo "COMP_WORDS=(${COMP_WORDS[@]})"
echo "COMP_CWORD=$COMP_CWORD"
echo "Number of words: ${#COMP_WORDS[@]}"
echo "Accessing COMP_WORDS[COMP_CWORD]: ${COMP_WORDS[$COMP_CWORD]}"
echo ""

echo -e "\nKey insight:"
echo "- COMP_CWORD is 0-indexed (starts from 0)"
echo "- COMP_CWORD=0 points to the command itself"
echo "- COMP_CWORD=1 points to the first argument"
echo "- When completing after the command, COMP_CWORD >= 1"