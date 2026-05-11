"""Build the Study Buddy fine-tuning dataset (JSONL, Alpaca-style).

Each record: {"instruction": str, "input": str, "output": str, "category": str}
  - instruction : the student's request
  - input       : optional context (empty string when not needed)
  - output      : the ideal Study Buddy response

Categories
  concept_explanation  – explain a topic clearly and simply
  flashcard_generation – produce Q/A flashcard sets
  practice_question    – generate practice problems with worked answers
  study_skills         – advice on how to study, revise, manage time
  exam_prep            – exam technique and worked exam-style answers

Output: data/processed/study_buddy_finetune.jsonl

Run from project root:
    python src/prepare_finetune_data.py
"""

import json
from pathlib import Path

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
OUT = PROCESSED / "study_buddy_finetune.jsonl"

# ── dataset ────────────────────────────────────────────────────────────────────

EXAMPLES = [

    # ── concept_explanation ────────────────────────────────────────────────────

    {
        "instruction": "Explain Newton's first law of motion in simple terms.",
        "input": "",
        "output": (
            "Newton's first law says that an object keeps doing whatever it is "
            "already doing — resting or moving — unless a force acts on it. "
            "This tendency is called inertia.\n\n"
            "Example: a book sitting on a desk won't slide unless you push it. "
            "A ball rolling across a perfectly smooth floor would roll forever "
            "because nothing is there to slow it down.\n\n"
            "In short: forces change motion; without a force, motion (or rest) "
            "continues unchanged."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between mean, median, and mode?",
        "input": "",
        "output": (
            "All three are measures of the 'centre' of a dataset, but they "
            "capture it differently.\n\n"
            "**Mean** – the arithmetic average. Add all values, divide by the "
            "count. Sensitive to outliers.\n"
            "  Example: [2, 4, 6, 100] → mean = 28\n\n"
            "**Median** – the middle value when the data is sorted. Robust to "
            "outliers.\n"
            "  Example: [2, 4, 6, 100] → median = (4+6)/2 = 5\n\n"
            "**Mode** – the most frequently occurring value. Useful for "
            "categorical data.\n"
            "  Example: [2, 4, 4, 6] → mode = 4\n\n"
            "Rule of thumb: use the median when outliers are present; the mean "
            "when data is roughly symmetric."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what a p-value is in statistics.",
        "input": "",
        "output": (
            "A p-value measures how surprising your data would be if the null "
            "hypothesis were true.\n\n"
            "More precisely: it is the probability of observing a result at "
            "least as extreme as the one you got, assuming the null hypothesis "
            "holds.\n\n"
            "Common threshold: p < 0.05. This means there is less than a 5 % "
            "chance the result is due to random chance alone, so we reject the "
            "null hypothesis.\n\n"
            "Important: a small p-value does NOT prove the alternative "
            "hypothesis is true — it only says the data is inconsistent with "
            "the null. p-values say nothing about effect size or practical "
            "importance."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is Big O notation and why does it matter?",
        "input": "",
        "output": (
            "Big O notation describes how an algorithm's run time (or memory "
            "use) grows as the input size n increases.\n\n"
            "Common complexities, from fastest to slowest:\n"
            "  O(1)      – constant: doesn't depend on n (e.g. array lookup)\n"
            "  O(log n)  – logarithmic: halves the problem each step (binary "
            "search)\n"
            "  O(n)      – linear: one pass through the data\n"
            "  O(n log n)– typical sorting algorithms (merge sort)\n"
            "  O(n²)     – quadratic: nested loops over n (bubble sort)\n"
            "  O(2ⁿ)     – exponential: blows up quickly (naive recursion)\n\n"
            "Why it matters: an O(n²) algorithm on 1 million items runs roughly "
            "1 trillion operations. An O(n log n) algorithm runs about 20 "
            "million. Choosing the right algorithm makes the difference between "
            "seconds and hours."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain the difference between supervised and unsupervised learning.",
        "input": "",
        "output": (
            "**Supervised learning** trains a model on labelled data — each "
            "example has an input and a known correct output. The model learns "
            "to map inputs to outputs.\n"
            "  Examples: spam detection (email → spam/not spam), house price "
            "prediction.\n\n"
            "**Unsupervised learning** works with unlabelled data. The model "
            "finds hidden structure on its own — no correct answers provided.\n"
            "  Examples: customer segmentation (k-means clustering), anomaly "
            "detection, topic modelling.\n\n"
            "Key difference: supervised needs labels (expensive to collect); "
            "unsupervised discovers patterns without them."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is gradient descent and how does it work?",
        "input": "",
        "output": (
            "Gradient descent is an optimisation algorithm used to minimise a "
            "loss function by iteratively adjusting model parameters.\n\n"
            "Steps:\n"
            "1. Start with random parameter values.\n"
            "2. Compute the loss (how wrong the model is).\n"
            "3. Calculate the gradient — the direction in which the loss "
            "increases most steeply.\n"
            "4. Move the parameters a small step in the opposite direction "
            "(downhill).\n"
            "5. Repeat until the loss stops decreasing.\n\n"
            "The step size is called the learning rate. Too large → the "
            "algorithm overshoots the minimum. Too small → convergence is slow. "
            "Mini-batch gradient descent is the most common variant used in "
            "deep learning."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what DNA replication is.",
        "input": "",
        "output": (
            "DNA replication is the process by which a cell copies its DNA "
            "before dividing, so each daughter cell gets a complete genome.\n\n"
            "Key steps:\n"
            "1. **Unwinding** – the enzyme helicase unzips the double helix by "
            "breaking hydrogen bonds between base pairs.\n"
            "2. **Priming** – primase lays down a short RNA primer to give DNA "
            "polymerase a starting point.\n"
            "3. **Synthesis** – DNA polymerase reads each template strand "
            "(3'→5') and builds the new complementary strand (5'→3') by adding "
            "nucleotides: A pairs with T, C pairs with G.\n"
            "4. **Proofreading** – DNA polymerase corrects most errors as it "
            "goes.\n"
            "5. **Sealing** – ligase joins any remaining gaps.\n\n"
            "Result: two identical DNA molecules, each with one original strand "
            "and one new strand (semi-conservative replication)."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is opportunity cost in economics?",
        "input": "",
        "output": (
            "Opportunity cost is the value of the next-best alternative you "
            "give up when making a choice.\n\n"
            "Example: you have £10,000 and choose to start a business. The "
            "opportunity cost is whatever else you could have done with that "
            "money — say, investing it at 5 % interest (£500 per year foregone)."
            "\n\n"
            "It applies to time too: studying for an exam means NOT watching "
            "TV, sleeping, or working a part-time shift. All those foregone "
            "benefits are the opportunity cost of studying.\n\n"
            "Why it matters: good decision-making requires comparing the full "
            "cost of a choice — not just the money paid, but also what you "
            "sacrifice."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain the central limit theorem simply.",
        "input": "",
        "output": (
            "The central limit theorem (CLT) says that if you take many "
            "independent random samples from any population — regardless of the "
            "population's shape — the distribution of the sample means will "
            "approach a normal (bell-curve) distribution as sample size grows.\n"
            "\n"
            "Why it matters:\n"
            "• It lets us use normal-distribution-based tests (t-tests, "
            "z-tests) even when the raw data is not normally distributed.\n"
            "• It explains why so many real-world averages look bell-shaped.\n\n"
            "Rule of thumb: with n ≥ 30, the sampling distribution of the mean "
            "is approximately normal for most populations."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between correlation and causation?",
        "input": "",
        "output": (
            "**Correlation** means two variables tend to move together — when "
            "one goes up, the other tends to go up (or down). It is a "
            "statistical relationship.\n\n"
            "**Causation** means one variable directly produces a change in "
            "the other.\n\n"
            "Classic example: ice cream sales and drowning rates are "
            "correlated. Does eating ice cream cause drowning? No — both rise "
            "in summer (confounding variable: warm weather).\n\n"
            "To establish causation you need:\n"
            "1. Correlation between the variables.\n"
            "2. The cause to come before the effect.\n"
            "3. No plausible alternative explanation (ideally a randomised "
            "controlled experiment).\n\n"
            "Short version: correlation is a clue, not proof of causation."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain recursion in programming with a simple example.",
        "input": "",
        "output": (
            "Recursion is when a function calls itself to solve a smaller "
            "version of the same problem.\n\n"
            "Every recursive function needs two things:\n"
            "1. **Base case** – a condition that stops the recursion.\n"
            "2. **Recursive case** – the function calling itself with a "
            "simpler input.\n\n"
            "Example — factorial:\n"
            "```python\n"
            "def factorial(n):\n"
            "    if n == 0:          # base case\n"
            "        return 1\n"
            "    return n * factorial(n - 1)   # recursive case\n"
            "```\n"
            "factorial(4) → 4 × factorial(3) → 4 × 3 × factorial(2) → … → "
            "4 × 3 × 2 × 1 = 24\n\n"
            "Tip: without a base case the function calls itself forever and "
            "causes a stack overflow."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between RAM and storage (hard drive)?",
        "input": "",
        "output": (
            "**RAM (Random Access Memory)**\n"
            "• Temporary, fast workspace used while programmes are running.\n"
            "• Contents are lost when power is turned off.\n"
            "• Typical sizes: 8–32 GB on a laptop.\n\n"
            "**Storage (HDD / SSD)**\n"
            "• Permanent memory that holds files, the OS, and applications.\n"
            "• Data survives when the computer is off.\n"
            "• Typical sizes: 256 GB – 2 TB.\n\n"
            "Analogy: RAM is your desk — you work on things there. Storage is "
            "the filing cabinet — things sit there until you bring them to "
            "your desk. More RAM means you can work on more things at once "
            "without slowing down."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what a SQL JOIN does.",
        "input": "",
        "output": (
            "A JOIN combines rows from two tables based on a matching column.\n"
            "\n"
            "Main types:\n"
            "**INNER JOIN** – returns only rows where the join condition "
            "matches in both tables.\n"
            "**LEFT JOIN** – returns all rows from the left table; matched rows "
            "from the right (NULLs where no match).\n"
            "**RIGHT JOIN** – opposite of LEFT JOIN.\n"
            "**FULL OUTER JOIN** – all rows from both tables; NULLs where "
            "either side has no match.\n\n"
            "Example:\n"
            "```sql\n"
            "SELECT students.name, grades.score\n"
            "FROM students\n"
            "INNER JOIN grades ON students.id = grades.student_id;\n"
            "```\n"
            "This returns only students who have a matching grade record. "
            "Students with no grade are excluded."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between TCP and UDP?",
        "input": "",
        "output": (
            "Both are transport-layer protocols for sending data over a "
            "network, but they trade reliability for speed differently.\n\n"
            "**TCP (Transmission Control Protocol)**\n"
            "• Connection-oriented: handshake before sending.\n"
            "• Guarantees delivery and correct order.\n"
            "• Slower due to acknowledgements and retransmissions.\n"
            "• Used for: web browsing (HTTP/S), email, file transfer.\n\n"
            "**UDP (User Datagram Protocol)**\n"
            "• Connectionless: just fires packets without confirmation.\n"
            "• No guarantee of delivery or order.\n"
            "• Fast and low-latency.\n"
            "• Used for: video streaming, online gaming, DNS, VoIP.\n\n"
            "Rule of thumb: use TCP when every byte must arrive correctly; "
            "use UDP when speed matters more than perfection."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what a confidence interval means.",
        "input": "",
        "output": (
            "A confidence interval (CI) gives a range of plausible values for "
            "an unknown population parameter, based on your sample.\n\n"
            "Example: a 95 % CI for a mean exam score of [68, 74] means:\n"
            "if we repeated the sampling process 100 times, about 95 of those "
            "intervals would contain the true population mean.\n\n"
            "Common misconception: it does NOT mean 'there is a 95 % chance "
            "the true mean is in this specific interval.' Once the interval is "
            "calculated, the true mean is either in it or it isn't.\n\n"
            "Wider CI = less precise (small sample or high variability). "
            "Narrower CI = more precise (large sample or low variability)."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is Ohm's law?",
        "input": "",
        "output": (
            "Ohm's law states that the current through a conductor is directly "
            "proportional to the voltage across it, provided temperature stays "
            "constant.\n\n"
            "Formula: **V = I × R**\n"
            "  V = voltage (volts, V)\n"
            "  I = current (amperes, A)\n"
            "  R = resistance (ohms, Ω)\n\n"
            "Rearranged:\n"
            "  I = V / R   (more voltage or less resistance → more current)\n"
            "  R = V / I\n\n"
            "Example: a 12 V battery connected to a 4 Ω resistor produces:\n"
            "  I = 12 / 4 = 3 A"
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between a list and a tuple in Python?",
        "input": "",
        "output": (
            "Both store ordered sequences of items, but:\n\n"
            "**List** `[]`\n"
            "• Mutable — you can add, remove, or change elements after "
            "creation.\n"
            "• Use when the collection will change.\n"
            "```python\n"
            "nums = [1, 2, 3]\n"
            "nums.append(4)   # [1, 2, 3, 4]\n"
            "```\n\n"
            "**Tuple** `()`\n"
            "• Immutable — once created it cannot be changed.\n"
            "• Slightly faster and can be used as dictionary keys.\n"
            "• Use for fixed collections like coordinates or RGB values.\n"
            "```python\n"
            "point = (10, 20)\n"
            "```\n\n"
            "Rule of thumb: if the data shouldn't change, use a tuple; "
            "otherwise use a list."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what photosynthesis is.",
        "input": "",
        "output": (
            "Photosynthesis is the process plants, algae, and some bacteria "
            "use to convert light energy into chemical energy stored as glucose."
            "\n\n"
            "Overall equation:\n"
            "  6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\n"
            "It happens in two stages:\n"
            "1. **Light-dependent reactions** (in thylakoids) – sunlight splits "
            "water molecules, releasing oxygen and producing ATP and NADPH.\n"
            "2. **Calvin cycle** (in stroma) – uses the ATP and NADPH to fix "
            "CO₂ from the air into glucose.\n\n"
            "Why it matters: photosynthesis is the entry point of almost all "
            "energy into food webs, and produces the oxygen we breathe."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between type I and type II errors?",
        "input": "",
        "output": (
            "Both are mistakes you can make when doing hypothesis testing.\n\n"
            "**Type I error (false positive)**\n"
            "Rejecting the null hypothesis when it is actually true.\n"
            "Example: concluding a drug works when it doesn't.\n"
            "Probability = α (significance level, often 0.05).\n\n"
            "**Type II error (false negative)**\n"
            "Failing to reject the null hypothesis when it is actually false.\n"
            "Example: concluding a drug has no effect when it actually does.\n"
            "Probability = β. Power = 1 − β.\n\n"
            "Trade-off: reducing α (stricter threshold) makes Type I errors "
            "rarer but Type II errors more common, and vice versa. Increasing "
            "sample size reduces both."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what entropy means in information theory.",
        "input": "",
        "output": (
            "In information theory, entropy measures the average amount of "
            "uncertainty (or information) in a random variable.\n\n"
            "Formula (Shannon entropy):\n"
            "  H(X) = −Σ p(x) log₂ p(x)\n\n"
            "Intuition:\n"
            "• A fair coin (50/50) has entropy = 1 bit — maximum uncertainty.\n"
            "• A biased coin (99 % heads) has entropy close to 0 — almost no "
            "surprise.\n"
            "• A uniform distribution over 8 outcomes has entropy = 3 bits.\n\n"
            "Used in: decision trees (information gain = entropy reduction), "
            "data compression, and evaluating language models."
        ),
        "category": "concept_explanation",
    },

    # ── flashcard_generation ───────────────────────────────────────────────────

    {
        "instruction": "Create 5 flashcards on the topic of SQL joins.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What does an INNER JOIN return?\n"
            "A: Only the rows where the join condition matches in both tables.\n"
            "\n"
            "**Flashcard 2**\n"
            "Q: What does a LEFT JOIN return?\n"
            "A: All rows from the left table; matching rows from the right. "
            "Non-matching right-side values are NULL.\n\n"
            "**Flashcard 3**\n"
            "Q: What is a FULL OUTER JOIN?\n"
            "A: Returns all rows from both tables. NULLs fill in wherever "
            "there is no match on either side.\n\n"
            "**Flashcard 4**\n"
            "Q: What is a CROSS JOIN?\n"
            "A: Returns every combination of rows from both tables "
            "(Cartesian product). No join condition needed.\n\n"
            "**Flashcard 5**\n"
            "Q: What is a SELF JOIN?\n"
            "A: A table joined to itself, typically to compare rows within "
            "the same table (e.g. finding employees who share a manager)."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Generate 6 flashcards on Newton's laws of motion.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: State Newton's first law.\n"
            "A: An object stays at rest or moves at constant velocity unless "
            "acted on by a net external force.\n\n"
            "**Flashcard 2**\n"
            "Q: What does Newton's first law define?\n"
            "A: Inertia — the resistance of an object to changes in its "
            "state of motion.\n\n"
            "**Flashcard 3**\n"
            "Q: State Newton's second law.\n"
            "A: F = ma. The net force on an object equals its mass times its "
            "acceleration.\n\n"
            "**Flashcard 4**\n"
            "Q: If mass doubles and force stays the same, what happens to "
            "acceleration?\n"
            "A: Acceleration halves (a = F/m).\n\n"
            "**Flashcard 5**\n"
            "Q: State Newton's third law.\n"
            "A: For every action there is an equal and opposite reaction.\n\n"
            "**Flashcard 6**\n"
            "Q: Give a real-world example of Newton's third law.\n"
            "A: A rocket expels gas downwards (action); the gas pushes the "
            "rocket upwards (reaction)."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Make 5 flashcards about Python data structures.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is a Python list?\n"
            "A: An ordered, mutable collection. Allows duplicates. "
            "Syntax: `[1, 2, 3]`.\n\n"
            "**Flashcard 2**\n"
            "Q: What is a Python tuple?\n"
            "A: An ordered, immutable collection. Faster than a list and "
            "hashable. Syntax: `(1, 2, 3)`.\n\n"
            "**Flashcard 3**\n"
            "Q: What is a Python dictionary?\n"
            "A: An unordered collection of key-value pairs. Keys must be "
            "unique and hashable. Syntax: `{'a': 1, 'b': 2}`.\n\n"
            "**Flashcard 4**\n"
            "Q: What is a Python set?\n"
            "A: An unordered collection of unique elements. Supports "
            "union, intersection, and difference operations. Syntax: `{1, 2, 3}`."
            "\n\n"
            "**Flashcard 5**\n"
            "Q: When would you use a deque instead of a list?\n"
            "A: When you need fast O(1) appends and pops from both ends. "
            "`collections.deque` is ideal for queues and sliding windows."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards on the topic of probability.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is the probability of an impossible event?\n"
            "A: 0.\n\n"
            "**Flashcard 2**\n"
            "Q: What is the probability of a certain event?\n"
            "A: 1.\n\n"
            "**Flashcard 3**\n"
            "Q: State the addition rule for mutually exclusive events.\n"
            "A: P(A or B) = P(A) + P(B).\n\n"
            "**Flashcard 4**\n"
            "Q: State the multiplication rule for independent events.\n"
            "A: P(A and B) = P(A) × P(B).\n\n"
            "**Flashcard 5**\n"
            "Q: What is conditional probability P(A|B)?\n"
            "A: The probability of A occurring given that B has already "
            "occurred. P(A|B) = P(A and B) / P(B)."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Generate 6 flashcards on machine learning model evaluation metrics.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What does accuracy measure?\n"
            "A: The proportion of all predictions that are correct: "
            "(TP + TN) / total.\n\n"
            "**Flashcard 2**\n"
            "Q: What is precision?\n"
            "A: Of all predicted positives, how many are actually positive: "
            "TP / (TP + FP).\n\n"
            "**Flashcard 3**\n"
            "Q: What is recall (sensitivity)?\n"
            "A: Of all actual positives, how many were correctly identified: "
            "TP / (TP + FN).\n\n"
            "**Flashcard 4**\n"
            "Q: What is the F1-score?\n"
            "A: The harmonic mean of precision and recall: "
            "2 × (precision × recall) / (precision + recall).\n\n"
            "**Flashcard 5**\n"
            "Q: What does ROC-AUC measure?\n"
            "A: The ability of a model to distinguish between classes across "
            "all classification thresholds. 1.0 = perfect, 0.5 = random.\n\n"
            "**Flashcard 6**\n"
            "Q: When should you prefer F1 over accuracy?\n"
            "A: When the dataset is imbalanced — accuracy can be misleadingly "
            "high if the model just predicts the majority class."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards on the human digestive system.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: Where does digestion begin?\n"
            "A: In the mouth — mechanical breakdown by teeth and chemical "
            "breakdown by salivary amylase.\n\n"
            "**Flashcard 2**\n"
            "Q: What is the role of the stomach?\n"
            "A: Churns food into chyme; secretes hydrochloric acid and pepsin "
            "to begin protein digestion.\n\n"
            "**Flashcard 3**\n"
            "Q: Where is most absorption of nutrients absorbed?\n"
            "A: The small intestine — villi and microvilli maximise surface "
            "area for absorption.\n\n"
            "**Flashcard 4**\n"
            "Q: What does the large intestine do?\n"
            "A: Absorbs water and electrolytes; compacts waste into faeces.\n"
            "\n"
            "**Flashcard 5**\n"
            "Q: What enzyme breaks down starch?\n"
            "A: Amylase (produced in the salivary glands and pancreas)."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Make 5 flashcards on key economic concepts.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is scarcity in economics?\n"
            "A: The fundamental problem that resources are limited while human "
            "wants are unlimited.\n\n"
            "**Flashcard 2**\n"
            "Q: What is the law of demand?\n"
            "A: As price rises, quantity demanded falls (inverse relationship), "
            "all else equal.\n\n"
            "**Flashcard 3**\n"
            "Q: What is the law of supply?\n"
            "A: As price rises, quantity supplied rises (positive relationship), "
            "all else equal.\n\n"
            "**Flashcard 4**\n"
            "Q: What is GDP?\n"
            "A: Gross Domestic Product — the total market value of all goods "
            "and services produced in a country in a given period.\n\n"
            "**Flashcard 5**\n"
            "Q: What is inflation?\n"
            "A: A sustained rise in the general price level of goods and "
            "services, reducing purchasing power."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards about the OSI model.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: How many layers does the OSI model have?\n"
            "A: 7 layers.\n\n"
            "**Flashcard 2**\n"
            "Q: Name the 7 OSI layers in order (top to bottom).\n"
            "A: Application, Presentation, Session, Transport, Network, "
            "Data Link, Physical.\n\n"
            "**Flashcard 3**\n"
            "Q: Which layer handles IP addressing and routing?\n"
            "A: Layer 3 — Network layer.\n\n"
            "**Flashcard 4**\n"
            "Q: Which layer handles end-to-end communication and TCP/UDP?\n"
            "A: Layer 4 — Transport layer.\n\n"
            "**Flashcard 5**\n"
            "Q: What is the role of the Physical layer (Layer 1)?\n"
            "A: Transmits raw bits over a physical medium (cables, radio "
            "waves, fibre optics)."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Generate 5 flashcards on key concepts in psychology.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is classical conditioning?\n"
            "A: Learning by association — a neutral stimulus is paired with "
            "an unconditioned stimulus until it produces the same response "
            "on its own (Pavlov's dogs).\n\n"
            "**Flashcard 2**\n"
            "Q: What is operant conditioning?\n"
            "A: Learning through consequences — behaviour is strengthened "
            "by reinforcement and weakened by punishment (Skinner).\n\n"
            "**Flashcard 3**\n"
            "Q: What is cognitive dissonance?\n"
            "A: The discomfort felt when holding two contradictory beliefs or "
            "when behaviour conflicts with beliefs.\n\n"
            "**Flashcard 4**\n"
            "Q: What are the stages of Piaget's cognitive development?\n"
            "A: Sensorimotor (0–2), Preoperational (2–7), Concrete operational "
            "(7–11), Formal operational (12+).\n\n"
            "**Flashcard 5**\n"
            "Q: What is the difference between short-term and long-term memory?"
            "\n"
            "A: Short-term (working) memory holds ~7 items for seconds to "
            "minutes. Long-term memory stores information indefinitely with "
            "potentially unlimited capacity."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards on graph algorithms.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is BFS (Breadth-First Search)?\n"
            "A: Explores nodes level by level using a queue. Finds the shortest "
            "path in unweighted graphs. Time: O(V + E).\n\n"
            "**Flashcard 2**\n"
            "Q: What is DFS (Depth-First Search)?\n"
            "A: Explores as far as possible along each branch before "
            "backtracking. Uses a stack (or recursion). Time: O(V + E).\n\n"
            "**Flashcard 3**\n"
            "Q: What does Dijkstra's algorithm find?\n"
            "A: The shortest path from a source node to all other nodes in "
            "a graph with non-negative edge weights.\n\n"
            "**Flashcard 4**\n"
            "Q: What is a minimum spanning tree?\n"
            "A: A subset of edges that connects all vertices with the minimum "
            "total weight and no cycles. Found by Kruskal's or Prim's.\n\n"
            "**Flashcard 5**\n"
            "Q: What is topological sorting?\n"
            "A: Linear ordering of vertices in a directed acyclic graph (DAG) "
            "such that for every edge u→v, u appears before v."
        ),
        "category": "flashcard_generation",
    },

    # ── practice_question ──────────────────────────────────────────────────────

    {
        "instruction": "Give me 3 practice problems on probability with solutions.",
        "input": "",
        "output": (
            "**Problem 1**\n"
            "A bag contains 3 red and 5 blue balls. What is the probability of "
            "drawing a red ball?\n"
            "Solution: P(red) = 3 / (3+5) = 3/8 = 0.375\n\n"
            "**Problem 2**\n"
            "Two fair dice are rolled. What is the probability that both show a "
            "6?\n"
            "Solution: P(6 on die 1) = 1/6, P(6 on die 2) = 1/6. "
            "Independent events: P(both 6) = 1/6 × 1/6 = 1/36 ≈ 0.028\n\n"
            "**Problem 3**\n"
            "A test is 90 % accurate. 1 % of the population has a disease. "
            "If someone tests positive, what is the probability they have the "
            "disease? (Bayes' theorem)\n"
            "Solution:\n"
            "  P(disease) = 0.01,  P(no disease) = 0.99\n"
            "  P(+|disease) = 0.90,  P(+|no disease) = 0.10\n"
            "  P(+) = 0.01×0.90 + 0.99×0.10 = 0.009 + 0.099 = 0.108\n"
            "  P(disease|+) = (0.01×0.90) / 0.108 ≈ 0.083 (about 8.3 %)"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 Python coding challenges with solutions.",
        "input": "",
        "output": (
            "**Challenge 1 — Reverse a string**\n"
            "Write a function that returns a string reversed.\n"
            "```python\n"
            "def reverse_string(s):\n"
            "    return s[::-1]\n"
            "print(reverse_string('hello'))  # 'olleh'\n"
            "```\n\n"
            "**Challenge 2 — Check palindrome**\n"
            "Return True if a string reads the same forwards and backwards.\n"
            "```python\n"
            "def is_palindrome(s):\n"
            "    s = s.lower().replace(' ', '')\n"
            "    return s == s[::-1]\n"
            "print(is_palindrome('racecar'))  # True\n"
            "```\n\n"
            "**Challenge 3 — FizzBuzz**\n"
            "Print 1–20. For multiples of 3 print 'Fizz', multiples of 5 "
            "print 'Buzz', both print 'FizzBuzz'.\n"
            "```python\n"
            "for i in range(1, 21):\n"
            "    if i % 15 == 0:\n"
            "        print('FizzBuzz')\n"
            "    elif i % 3 == 0:\n"
            "        print('Fizz')\n"
            "    elif i % 5 == 0:\n"
            "        print('Buzz')\n"
            "    else:\n"
            "        print(i)\n"
            "```"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice questions on Newton's laws with answers.",
        "input": "",
        "output": (
            "**Question 1**\n"
            "A 10 kg box is pushed with a net force of 30 N. What is its "
            "acceleration?\n"
            "Answer: F = ma → a = F/m = 30/10 = 3 m/s²\n\n"
            "**Question 2**\n"
            "A skater pushes off a wall with 50 N of force. What force does "
            "the wall exert on the skater?\n"
            "Answer: 50 N in the opposite direction (Newton's third law). "
            "The wall pushes back with equal force.\n\n"
            "**Question 3**\n"
            "A car travelling at 20 m/s brakes and stops in 4 s. The car "
            "has mass 1,200 kg. What braking force was applied?\n"
            "Answer: a = Δv/t = (0−20)/4 = −5 m/s². "
            "F = ma = 1,200 × (−5) = −6,000 N (6 kN opposing motion)."
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Create 3 practice questions on SQL with solutions.",
        "input": "",
        "output": (
            "**Question 1**\n"
            "Write a query to find all students with a score above 80, "
            "sorted by score descending.\n"
            "```sql\n"
            "SELECT * FROM students\n"
            "WHERE score > 80\n"
            "ORDER BY score DESC;\n"
            "```\n\n"
            "**Question 2**\n"
            "Count how many students are in each module.\n"
            "```sql\n"
            "SELECT module_id, COUNT(*) AS student_count\n"
            "FROM enrolments\n"
            "GROUP BY module_id;\n"
            "```\n\n"
            "**Question 3**\n"
            "Find students who have never submitted an assessment (using a "
            "LEFT JOIN).\n"
            "```sql\n"
            "SELECT s.student_id, s.name\n"
            "FROM students s\n"
            "LEFT JOIN submissions sub ON s.student_id = sub.student_id\n"
            "WHERE sub.student_id IS NULL;\n"
            "```"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice questions on hypothesis testing.",
        "input": "",
        "output": (
            "**Question 1**\n"
            "A coin is flipped 100 times and lands heads 60 times. At α = 0.05, "
            "is there evidence the coin is biased?\n"
            "Answer: H₀: p = 0.5. z = (0.60 − 0.50) / √(0.5×0.5/100) = "
            "0.10/0.05 = 2.0. Critical value at α=0.05 (two-tailed) ≈ 1.96. "
            "Since 2.0 > 1.96, reject H₀ — evidence of bias.\n\n"
            "**Question 2**\n"
            "Explain what it means to fail to reject the null hypothesis.\n"
            "Answer: It means the data does not provide enough evidence to "
            "conclude the null is false. It does NOT prove the null is true — "
            "absence of evidence is not evidence of absence.\n\n"
            "**Question 3**\n"
            "Increasing sample size from 50 to 500 — how does this affect "
            "Type I and Type II errors?\n"
            "Answer: Type I error (α) is set by the researcher and stays the "
            "same. Type II error (β) decreases — the test has more power to "
            "detect real effects."
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice problems on linear algebra — matrix operations.",
        "input": "",
        "output": (
            "**Problem 1 — Matrix addition**\n"
            "A = [[1,2],[3,4]], B = [[5,6],[7,8]]. Find A + B.\n"
            "Answer: A + B = [[1+5, 2+6],[3+7, 4+8]] = [[6,8],[10,12]]\n\n"
            "**Problem 2 — Matrix multiplication**\n"
            "A = [[1,2],[3,4]], B = [[5,6],[7,8]]. Find AB.\n"
            "Answer:\n"
            "  Row 1 of A × Col 1 of B: 1×5 + 2×7 = 19\n"
            "  Row 1 of A × Col 2 of B: 1×6 + 2×8 = 22\n"
            "  Row 2 of A × Col 1 of B: 3×5 + 4×7 = 43\n"
            "  Row 2 of A × Col 2 of B: 3×6 + 4×8 = 50\n"
            "  AB = [[19,22],[43,50]]\n\n"
            "**Problem 3 — Determinant**\n"
            "Find the determinant of [[3,8],[4,6]].\n"
            "Answer: det = (3×6) − (8×4) = 18 − 32 = −14"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice questions on object-oriented programming concepts.",
        "input": "",
        "output": (
            "**Question 1**\n"
            "What is the difference between a class and an object?\n"
            "Answer: A class is a blueprint defining attributes and behaviour. "
            "An object is an instance of a class — actual data in memory. "
            "Example: `Car` is a class; `my_car = Car('Toyota')` creates an "
            "object.\n\n"
            "**Question 2**\n"
            "What is inheritance and why use it?\n"
            "Answer: A child class inherits attributes and methods from a "
            "parent class, allowing code reuse. Example: `Dog` and `Cat` both "
            "inherit from `Animal` and share a `breathe()` method.\n\n"
            "**Question 3**\n"
            "What is polymorphism? Give a one-line code example.\n"
            "Answer: Different classes responding to the same method call in "
            "their own way.\n"
            "```python\n"
            "for animal in [Dog(), Cat()]:\n"
            "    animal.speak()  # Dog barks, Cat meows\n"
            "```"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Provide 3 short-answer questions on the topic of climate change with answers.",
        "input": "",
        "output": (
            "**Question 1**\n"
            "What is the greenhouse effect?\n"
            "Answer: Solar radiation passes through the atmosphere and warms "
            "Earth's surface. The surface re-emits heat as infrared radiation, "
            "which greenhouse gases (CO₂, CH₄, H₂O) trap, preventing it from "
            "escaping to space. This naturally warms the planet; human "
            "emissions amplify the effect.\n\n"
            "**Question 2**\n"
            "Name two major consequences of global warming.\n"
            "Answer: (1) Rising sea levels due to ice sheet melt and thermal "
            "expansion of oceans. (2) More frequent and intense extreme weather "
            "events (heatwaves, hurricanes, floods).\n\n"
            "**Question 3**\n"
            "What is the difference between mitigation and adaptation in "
            "climate policy?\n"
            "Answer: Mitigation reduces greenhouse gas emissions to slow "
            "climate change (e.g. renewable energy). Adaptation adjusts "
            "systems to cope with changes already occurring "
            "(e.g. sea walls, drought-resistant crops)."
        ),
        "category": "practice_question",
    },

    # ── study_skills ───────────────────────────────────────────────────────────

    {
        "instruction": "What is the Pomodoro technique and how do I use it for studying?",
        "input": "",
        "output": (
            "The Pomodoro technique breaks work into focused intervals "
            "separated by short breaks, helping maintain concentration and "
            "avoid burnout.\n\n"
            "How to use it:\n"
            "1. Choose a task to work on.\n"
            "2. Set a timer for 25 minutes and work without distractions.\n"
            "3. When the timer rings, take a 5-minute break.\n"
            "4. After 4 Pomodoros, take a longer break (15–30 minutes).\n\n"
            "Tips:\n"
            "• Put your phone on Do Not Disturb during the 25 minutes.\n"
            "• Write down any distracting thoughts on paper to deal with later.\n"
            "• Adjust interval length if 25 min is too short or too long "
            "— some people prefer 50/10."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "How do I use spaced repetition to remember more of what I study?",
        "input": "",
        "output": (
            "Spaced repetition exploits the 'spacing effect': reviewing "
            "material at increasing intervals just before you are about to "
            "forget it strengthens long-term memory more efficiently than "
            "cramming.\n\n"
            "How to do it:\n"
            "1. After first learning something, review it the next day.\n"
            "2. If you recall it well, review again in 3 days.\n"
            "3. Then 1 week, 2 weeks, 1 month, and so on.\n"
            "4. If you forget, reset the interval back to 1 day.\n\n"
            "Tools:\n"
            "• **Anki** (free) — automatically schedules cards based on how "
            "well you recalled them.\n"
            "• Physical flashcard box with dividers for each interval.\n\n"
            "Key principle: the harder the recall, the stronger the memory "
            "formed. Don't review things you already know well — focus on "
            "the difficult ones."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "I have 3 weeks to prepare for a difficult exam. How should I plan?",
        "input": "",
        "output": (
            "Here is a structured 3-week plan:\n\n"
            "**Week 1 — Content review**\n"
            "• List every topic on the syllabus.\n"
            "• Work through each topic once, taking concise notes.\n"
            "• Flag anything you don't understand for closer attention.\n\n"
            "**Week 2 — Active recall**\n"
            "• Close your notes and test yourself on each topic.\n"
            "• Use flashcards or practice problems — not re-reading.\n"
            "• Focus extra time on flagged weak areas.\n\n"
            "**Week 3 — Past papers and consolidation**\n"
            "• Complete past or practice exams under timed conditions.\n"
            "• Review every wrong answer and understand why it was wrong.\n"
            "• Light review the day before — avoid cramming new material.\n\n"
            "General tips:\n"
            "• Study in 1–2 hour sessions with breaks.\n"
            "• Sleep is critical — memory consolidation happens during sleep.\n"
            "• Start early; don't compress week 1 into the last 2 days."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "What is active recall and why is it better than re-reading notes?",
        "input": "",
        "output": (
            "Active recall means forcing yourself to retrieve information from "
            "memory, rather than passively reading it again.\n\n"
            "Why it works better:\n"
            "• Retrieval practice strengthens the memory trace far more than "
            "re-exposure (the 'testing effect', well-supported by research).\n"
            "• It reveals gaps in your knowledge — you can't fool yourself "
            "that you know something when you have to produce it.\n"
            "• Re-reading creates 'fluency illusion': the material feels "
            "familiar so you think you know it, but recognition ≠ recall.\n\n"
            "How to practise active recall:\n"
            "• Cover your notes and write down everything you remember.\n"
            "• Answer flashcard questions without looking at the answer first.\n"
            "• Explain the concept aloud as if teaching someone else "
            "(the Feynman technique).\n"
            "• Do practice problems from memory."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "How do I take better notes during lectures?",
        "input": "",
        "output": (
            "Effective lecture notes capture ideas, not transcripts. "
            "Trying to write everything down is slower and less useful than "
            "engaging with the content.\n\n"
            "Techniques:\n\n"
            "**Cornell method** — divide the page into notes (right), cues "
            "(left, added after), and summary (bottom). After the lecture, "
            "fill in cues and write a 2–3 sentence summary.\n\n"
            "**Mind maps** — good for conceptual topics. Write the main idea "
            "in the centre, branch out to sub-topics.\n\n"
            "**Outline method** — structured bullet points with indentation "
            "for sub-points. Clear hierarchy.\n\n"
            "General advice:\n"
            "• Write in your own words — this forces understanding.\n"
            "• Leave white space to add detail later.\n"
            "• Mark unclear points with '?' to revisit.\n"
            "• Review and tidy notes within 24 hours while memory is fresh."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "I keep procrastinating on my assignments. What can I do?",
        "input": "",
        "output": (
            "Procrastination usually comes from anxiety, perfectionism, or "
            "the task feeling overwhelming — not laziness.\n\n"
            "Strategies that work:\n\n"
            "**2-minute rule** — if starting feels hard, commit to just "
            "2 minutes. Getting started is usually the hardest part.\n\n"
            "**Break it down** — instead of 'write essay', write:\n"
            "  1. Write the question at the top of a blank page.\n"
            "  2. Find 3 sources.\n"
            "  3. Write a 3-point outline.\n"
            "Small, concrete next actions are much easier to begin.\n\n"
            "**Remove friction** — close social media, use a site blocker "
            "(Freedom, Cold Turkey), keep your phone in another room.\n\n"
            "**Change environment** — a library or quiet café often works "
            "better than your room.\n\n"
            "**Done > perfect** — write a rough draft first. Editing an "
            "imperfect draft is far easier than staring at a blank page."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "What is the Feynman technique and how do I use it?",
        "input": "",
        "output": (
            "The Feynman technique is a four-step method for learning anything "
            "deeply by testing whether you can explain it simply.\n\n"
            "Steps:\n"
            "1. **Choose a concept** and write its name at the top of a page.\n"
            "2. **Explain it in plain language** as if teaching a 12-year-old — "
            "no jargon.\n"
            "3. **Identify gaps** — wherever your explanation breaks down or "
            "feels vague, go back to the source material and study that part.\n"
            "4. **Simplify further** — replace any jargon or overly complex "
            "phrasing with simpler analogies.\n\n"
            "Why it works: if you cannot explain something simply, you don't "
            "truly understand it yet. The act of explaining forces you to "
            "confront gaps that passive reading hides."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "How much sleep do I need and how does sleep affect studying?",
        "input": "",
        "output": (
            "Most adults need 7–9 hours of sleep per night. Students tend to "
            "underestimate how much sleep affects academic performance.\n\n"
            "What sleep does for learning:\n"
            "• **Memory consolidation** — during sleep (especially deep sleep "
            "and REM), the brain replays and strengthens what you studied.\n"
            "• **Clearing waste** — the glymphatic system removes metabolic "
            "waste from the brain, including amyloid plaques.\n"
            "• **Focus and attention** — just one night of poor sleep "
            "significantly impairs concentration, working memory, and "
            "problem-solving.\n\n"
            "Practical tips:\n"
            "• Keep a consistent sleep schedule — same wake time every day.\n"
            "• Avoid screens for 30–60 min before bed (blue light delays "
            "melatonin).\n"
            "• Don't pull all-nighters before exams — a well-rested brain "
            "outperforms a sleep-deprived one even with less study time."
        ),
        "category": "study_skills",
    },

    # ── exam_prep ─────────────────────────────────────────────────────────────

    {
        "instruction": "How should I approach a multiple-choice exam?",
        "input": "",
        "output": (
            "Multiple-choice exams reward systematic technique as much as "
            "knowledge.\n\n"
            "Strategy:\n"
            "1. **First pass** — answer every question you know confidently. "
            "Skip uncertain ones and mark them.\n"
            "2. **Elimination** — on marked questions, rule out obviously "
            "wrong answers. If you can eliminate 2 of 4, your guessing odds "
            "improve from 25 % to 50 %.\n"
            "3. **Read carefully** — watch for qualifiers like 'always', "
            "'never', 'most likely'. These often make an answer wrong.\n"
            "4. **Consistent answers** — if two questions contain clues for "
            "each other, use them.\n"
            "5. **First instinct** — research shows changing an answer on "
            "reflection is right about as often as wrong. Only change if you "
            "have a clear reason.\n"
            "6. **Don't leave blanks** — unless there is a penalty for wrong "
            "answers, always guess."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How do I write a strong essay under timed exam conditions?",
        "input": "",
        "output": (
            "Timed essays reward planning and clear structure over length.\n\n"
            "Framework (for a 45-minute essay):\n"
            "• 5 min — plan: jot down your thesis and 3 main points.\n"
            "• 35 min — write: intro, 3 body paragraphs, conclusion.\n"
            "• 5 min — review: check clarity, fix obvious errors.\n\n"
            "Paragraph structure (PEEL):\n"
            "  **P**oint — state your argument.\n"
            "  **E**vidence — support with data, example, or theory.\n"
            "  **E**xplain — connect the evidence to your point.\n"
            "  **L**ink — connect back to the essay question.\n\n"
            "Tips:\n"
            "• Answer the question directly in the first sentence — markers "
            "reward students who stay on topic.\n"
            "• A clear 3-paragraph essay beats a sprawling 6-paragraph one.\n"
            "• Leave 2–3 lines between paragraphs if you want to add points "
            "in your review pass."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "What should I do the night before and morning of an exam?",
        "input": "",
        "output": (
            "**Night before**\n"
            "• Do a light review (30–60 min max) — summaries and key "
            "formulas only. No new material.\n"
            "• Prepare everything you need: ID, pens, calculator, water.\n"
            "• Eat a proper meal.\n"
            "• Go to bed at your normal time — sacrificing sleep for extra "
            "study is counterproductive.\n\n"
            "**Morning of**\n"
            "• Eat breakfast — your brain needs glucose.\n"
            "• Arrive early enough to settle without rushing.\n"
            "• Do a quick, calm recap of key facts — avoid panicked cramming.\n"
            "• Avoid comparing notes with classmates outside the exam room; "
            "it raises anxiety without helping.\n\n"
            "**In the exam**\n"
            "• Read all instructions before starting.\n"
            "• Allocate time per question based on marks.\n"
            "• If you freeze, move on and come back."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How do I manage exam anxiety?",
        "input": "",
        "output": (
            "Some anxiety is normal and even helpful — it sharpens focus. "
            "The goal is to keep it manageable.\n\n"
            "Before the exam:\n"
            "• **Preparation reduces anxiety** — consistent study over weeks "
            "builds genuine confidence.\n"
            "• Visualise succeeding, not failing.\n"
            "• Avoid caffeine overload the day before.\n\n"
            "During the exam:\n"
            "• **Box breathing** — inhale 4 s, hold 4 s, exhale 4 s, hold 4 s. "
            "Repeat 3 times. This activates the parasympathetic nervous system.\n"
            "• If you go blank on a question, skip it and return. "
            "Momentum from answering easier questions often unlocks memory.\n"
            "• Remind yourself: the anxiety feeling is your body preparing "
            "to perform, not failing.\n\n"
            "After the exam:\n"
            "• Don't re-hash every question with classmates — it serves no "
            "purpose and increases stress."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How should I review a graded exam to improve for next time?",
        "input": "",
        "output": (
            "Post-exam review is one of the most effective (and most skipped) "
            "study activities.\n\n"
            "Steps:\n"
            "1. **Get your paper back** — if the examiner returns it, go "
            "through every question, not just the ones you got wrong.\n"
            "2. **Categorise errors**:\n"
            "   • Didn't know the content → add to revision list.\n"
            "   • Knew it but misread the question → practice careful "
            "reading.\n"
            "   • Made a careless arithmetic or logic error → slow down and "
            "check working in future.\n"
            "   • Ran out of time → practice timed conditions more.\n"
            "3. **Rework wrong answers** — don't just read the correct answer; "
            "redo the problem yourself.\n"
            "4. **Update your notes** — add anything the exam revealed you "
            "were missing.\n"
            "5. **Adjust your study method** — if you revised heavily but "
            "still failed application questions, you may be memorising rather "
            "than understanding."
        ),
        "category": "exam_prep",
    },
]


# ── write JSONL ────────────────────────────────────────────────────────────────

def main():
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(EXAMPLES)
    by_category: dict[str, int] = {}
    for ex in EXAMPLES:
        by_category[ex["category"]] = by_category.get(ex["category"], 0) + 1

    print(f"Wrote {total} examples → {OUT}\n")
    print("Breakdown by category:")
    for cat, n in sorted(by_category.items()):
        print(f"  {cat:<25} {n:>3}")

    # Basic token-length proxy (word count)
    lengths = [len(ex["output"].split()) for ex in EXAMPLES]
    print(f"\nOutput word count — min: {min(lengths)}  "
          f"median: {sorted(lengths)[len(lengths)//2]}  "
          f"max: {max(lengths)}")


if __name__ == "__main__":
    main()
