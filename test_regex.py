import re

prompt = "a multi-cloud Kubernetes deployment, incorporating automated testing, vulnerability scanning, and blue/green deployments."
prompt_lower = prompt.lower()

regex = r"a multi-cloud kubernetes deployment(?:.*?)?"

match = re.search(regex, prompt_lower)

if match:
    print("Regex matched!")
    print(f"Match: {match.group(0)}")
else:
    print("Regex did NOT match.")
