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

    # ── concept_explanation — batch 1 (original 20) ───────────────────────────

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

    # ── concept_explanation — batch 2 ─────────────────────────────────────────

    {
        "instruction": "How does a hash table work?",
        "input": "",
        "output": (
            "A hash table stores key-value pairs for near-instant lookup.\n\n"
            "How it works:\n"
            "1. A hash function converts the key into an integer index.\n"
            "2. The value is stored at that index in an underlying array.\n"
            "3. To retrieve a value, hash the key again and go directly to "
            "that index.\n\n"
            "Collisions (two keys mapping to the same index) are handled by "
            "chaining (linked list at that slot) or open addressing (probe "
            "for the next free slot).\n\n"
            "Average time complexity: O(1) for insert, lookup, and delete. "
            "Worst case (many collisions): O(n).\n\n"
            "Used in: Python dicts, database indexes, caches, sets."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between a stack and a queue?",
        "input": "",
        "output": (
            "Both are linear data structures but differ in the order items "
            "are removed.\n\n"
            "**Stack — LIFO (Last In, First Out)**\n"
            "The last item added is the first removed — like a stack of plates."
            "\n"
            "Operations: push (add to top), pop (remove from top).\n"
            "Uses: undo history, function call stack, bracket matching.\n\n"
            "**Queue — FIFO (First In, First Out)**\n"
            "The first item added is the first removed — like a queue at a "
            "ticket office.\n"
            "Operations: enqueue (add to back), dequeue (remove from front).\n"
            "Uses: task scheduling, BFS, print spooling.\n\n"
            "Python: use `collections.deque` for both; it supports O(1) "
            "operations at both ends."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what overfitting and underfitting mean in machine learning.",
        "input": "",
        "output": (
            "**Overfitting** — the model learns the training data too well, "
            "including its noise. It performs great on training data but poorly "
            "on new data. Signs: very high train accuracy, much lower test "
            "accuracy.\n\n"
            "**Underfitting** — the model is too simple to capture the "
            "underlying pattern. It performs poorly on both train and test "
            "data.\n\n"
            "Fixes for overfitting:\n"
            "• Regularisation (L1/L2)\n"
            "• More training data\n"
            "• Dropout (neural networks)\n"
            "• Simpler model\n\n"
            "Fixes for underfitting:\n"
            "• More complex model\n"
            "• More features\n"
            "• Train longer\n\n"
            "The goal is the sweet spot between the two — good generalisation."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is binary search and when should I use it?",
        "input": "",
        "output": (
            "Binary search finds a target value in a sorted array by repeatedly "
            "halving the search space.\n\n"
            "Steps:\n"
            "1. Look at the middle element.\n"
            "2. If it equals the target, done.\n"
            "3. If the target is smaller, search the left half.\n"
            "4. If larger, search the right half.\n"
            "5. Repeat until found or the space is empty.\n\n"
            "Time complexity: O(log n) — far faster than linear search O(n) "
            "for large arrays.\n\n"
            "Use it when: the data is sorted and you need repeated lookups.\n"
            "Don't use it when: data is unsorted (sort first, or use a hash "
            "table for O(1) lookups).\n\n"
            "Python: `bisect` module implements binary search."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is regularisation in machine learning and why does it help?",
        "input": "",
        "output": (
            "Regularisation adds a penalty to the loss function to discourage "
            "the model from fitting noise in the training data, reducing "
            "overfitting.\n\n"
            "**L1 (Lasso)** — penalises the sum of absolute weights. Drives "
            "some weights to exactly zero, performing feature selection.\n\n"
            "**L2 (Ridge)** — penalises the sum of squared weights. Shrinks "
            "weights toward zero without eliminating them. More common.\n\n"
            "**ElasticNet** — combines L1 and L2.\n\n"
            "The regularisation strength is controlled by a hyperparameter λ "
            "(or α in sklearn): higher λ = stronger penalty = simpler model.\n\n"
            "Rule of thumb: try L2 first; switch to L1 if you want automatic "
            "feature selection."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is a linked list and how is it different from an array?",
        "input": "",
        "output": (
            "A linked list stores elements as nodes, where each node holds a "
            "value and a pointer to the next node.\n\n"
            "**Array**\n"
            "• Contiguous memory — elements sit side by side.\n"
            "• O(1) random access by index.\n"
            "• Insertion/deletion in the middle: O(n) (must shift elements).\n"
            "• Fixed size (static) or expensive resize (dynamic).\n\n"
            "**Linked list**\n"
            "• Scattered memory — nodes linked by pointers.\n"
            "• O(n) access (must traverse from head).\n"
            "• O(1) insertion/deletion at head or tail.\n"
            "• Naturally dynamic size.\n\n"
            "Use arrays when you need fast lookups; use linked lists when you "
            "need frequent insertions and deletions at the ends."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is database normalisation?",
        "input": "",
        "output": (
            "Normalisation organises a database to reduce data redundancy and "
            "improve integrity. It is done in steps called Normal Forms (NF).\n"
            "\n"
            "**1NF** — each cell holds one value; no repeating groups.\n"
            "**2NF** — 1NF + every non-key column depends on the whole primary "
            "key (no partial dependencies).\n"
            "**3NF** — 2NF + no non-key column depends on another non-key "
            "column (no transitive dependencies).\n\n"
            "Example of a 3NF violation: storing `postcode` and `city` in the "
            "same table when city can be derived from postcode — move city to "
            "a separate Postcodes table.\n\n"
            "Most production databases aim for 3NF. "
            "Denormalisation (deliberately breaking NF) is sometimes used for "
            "read performance."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is an API and how does it work?",
        "input": "",
        "output": (
            "An API (Application Programming Interface) is a defined set of "
            "rules that lets two pieces of software communicate.\n\n"
            "Analogy: a restaurant menu is an API — it defines what you can "
            "order (requests) and what the kitchen will return (responses). "
            "You don't need to know how the kitchen works.\n\n"
            "A web API typically works over HTTP:\n"
            "1. Client sends a request (GET /users/42).\n"
            "2. Server processes it and returns a response (JSON with user "
            "data).\n\n"
            "Common methods:\n"
            "  GET    — retrieve data\n"
            "  POST   — create data\n"
            "  PUT    — update/replace data\n"
            "  DELETE — remove data\n\n"
            "REST APIs are stateless — each request contains all the "
            "information needed; the server stores no session."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is Git and why do developers use it?",
        "input": "",
        "output": (
            "Git is a distributed version control system — it tracks every "
            "change made to code over time, so you can review history, undo "
            "mistakes, and collaborate without overwriting each other's work."
            "\n\n"
            "Key concepts:\n"
            "**Repository (repo)** — the project folder Git is tracking.\n"
            "**Commit** — a saved snapshot of changes, with a message.\n"
            "**Branch** — a parallel version of the code for a feature or fix."
            "\n"
            "**Merge** — combining a branch back into the main codebase.\n"
            "**Remote** — a copy of the repo on a server (e.g. GitHub).\n\n"
            "Basic workflow:\n"
            "```\n"
            "git add .           # stage changes\n"
            "git commit -m '...' # save snapshot\n"
            "git push            # upload to remote\n"
            "```\n\n"
            "Why it matters: without version control, one mistake can destroy "
            "days of work. Git makes every change reversible."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "Explain what a neural network is.",
        "input": "",
        "output": (
            "A neural network is a machine learning model loosely inspired by "
            "the brain. It learns to map inputs to outputs by adjusting "
            "millions of numeric weights.\n\n"
            "Structure:\n"
            "• **Input layer** — receives raw features (pixels, numbers, text "
            "tokens).\n"
            "• **Hidden layers** — apply weighted sums + non-linear activation "
            "functions (ReLU, sigmoid) to extract patterns.\n"
            "• **Output layer** — produces predictions (class probabilities, "
            "a number, etc.).\n\n"
            "Training: feed data forward, compute loss, backpropagate the "
            "error gradient, update weights via gradient descent. Repeat for "
            "many epochs.\n\n"
            "Deep learning = networks with many hidden layers. Used for image "
            "recognition, NLP, speech, and more."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is cross-validation and why is it used?",
        "input": "",
        "output": (
            "Cross-validation evaluates a model's ability to generalise to "
            "unseen data more reliably than a single train/test split.\n\n"
            "Most common form — k-fold CV:\n"
            "1. Split data into k equal folds (typically k = 5 or 10).\n"
            "2. Train on k−1 folds, test on the remaining fold.\n"
            "3. Repeat k times, each fold serving as the test set once.\n"
            "4. Average the k test scores.\n\n"
            "Why use it:\n"
            "• A single split is sensitive to which samples land in test. "
            "CV averages this out.\n"
            "• Gives an honest estimate of real-world performance.\n"
            "• Essential for hyperparameter tuning — otherwise you overfit "
            "the test set.\n\n"
            "Stratified k-fold preserves the class distribution in each fold — "
            "always use it for classification."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between a compiled and an interpreted language?",
        "input": "",
        "output": (
            "**Compiled language** (C, C++, Go, Rust)\n"
            "Source code is translated entirely into machine code before "
            "running. The compiler catches errors at build time. Resulting "
            "binary runs fast — no translation overhead at runtime.\n\n"
            "**Interpreted language** (Python, JavaScript, Ruby)\n"
            "Source code is read and executed line by line by an interpreter "
            "at runtime. Errors appear when that line is reached. More "
            "flexible and faster to develop with, but generally slower "
            "to run.\n\n"
            "**JIT-compiled** (Java, C#, modern JS engines)\n"
            "A middle ground: code is compiled to bytecode, then the JIT "
            "compiler translates hot paths to machine code at runtime.\n\n"
            "For most student projects, interpreted languages are faster to "
            "write; compiled languages matter when raw performance is critical."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between a process and a thread?",
        "input": "",
        "output": (
            "**Process** — an independent programme in execution with its own "
            "memory space. Isolated from other processes. Expensive to create "
            "and communicate between.\n\n"
            "**Thread** — a lightweight unit of execution within a process. "
            "Threads share the same memory space, making communication fast "
            "but requiring careful synchronisation to avoid race conditions.\n"
            "\n"
            "Analogy: a process is a separate factory; threads are workers "
            "inside the same factory sharing the same tools.\n\n"
            "Python note: due to the GIL (Global Interpreter Lock), Python "
            "threads cannot run CPU-bound code in true parallel. Use "
            "`multiprocessing` for CPU work; `threading` or `asyncio` for "
            "I/O-bound tasks."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the derivative in calculus?",
        "input": "",
        "output": (
            "The derivative measures the instantaneous rate of change of a "
            "function — how steeply it rises or falls at a given point.\n\n"
            "Notation: f'(x), dy/dx, or df/dx.\n\n"
            "Geometrically: the derivative at a point is the slope of the "
            "tangent line to the curve at that point.\n\n"
            "Key rules:\n"
            "  Power rule:   d/dx [xⁿ] = nxⁿ⁻¹\n"
            "  Constant:     d/dx [c] = 0\n"
            "  Sum:          d/dx [f + g] = f' + g'\n"
            "  Product:      d/dx [fg] = f'g + fg'\n"
            "  Chain rule:   d/dx [f(g(x))] = f'(g(x)) · g'(x)\n\n"
            "Example: f(x) = 3x² → f'(x) = 6x\n\n"
            "Used in optimisation (set derivative to zero to find minima/maxima) "
            "and is the foundation of gradient descent in ML."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is natural selection?",
        "input": "",
        "output": (
            "Natural selection is the mechanism Darwin proposed to explain how "
            "species evolve over time.\n\n"
            "Four conditions must hold:\n"
            "1. **Variation** — individuals in a population differ in traits.\n"
            "2. **Heritability** — traits are passed from parents to offspring."
            "\n"
            "3. **Differential survival/reproduction** — some traits improve "
            "survival or reproduction in the current environment.\n"
            "4. **Selection pressure** — the environment 'selects' for "
            "advantageous traits.\n\n"
            "Over generations, advantageous traits become more common; "
            "harmful ones decline.\n\n"
            "Example: peppered moths in industrial England — dark moths were "
            "better camouflaged on soot-covered trees and so survived more, "
            "shifting the population darker."
        ),
        "category": "concept_explanation",
    },

    # ── concept_explanation — batch 3 ─────────────────────────────────────────

    {
        "instruction": "What is the law of supply and demand?",
        "input": "",
        "output": (
            "Supply and demand describes how prices and quantities are "
            "determined in a market.\n\n"
            "**Demand**: as price rises, consumers buy less (inverse "
            "relationship). The demand curve slopes downward.\n\n"
            "**Supply**: as price rises, producers supply more (direct "
            "relationship). The supply curve slopes upward.\n\n"
            "**Equilibrium**: the price where supply equals demand — the "
            "market clears with no surplus or shortage.\n\n"
            "Shifts:\n"
            "• Demand rises (curve shifts right) → price rises.\n"
            "• Supply rises (curve shifts right) → price falls.\n\n"
            "Real example: a drought reduces wheat supply → wheat price rises "
            "until consumers reduce demand to match the lower supply."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the scientific method?",
        "input": "",
        "output": (
            "The scientific method is a systematic process for testing ideas "
            "against evidence.\n\n"
            "Steps:\n"
            "1. **Observe** — notice a phenomenon or problem.\n"
            "2. **Question** — formulate a specific, answerable question.\n"
            "3. **Hypothesis** — make a testable prediction.\n"
            "4. **Experiment** — design a controlled test to check the "
            "prediction.\n"
            "5. **Data collection** — record results carefully.\n"
            "6. **Analysis** — look for patterns; use statistics.\n"
            "7. **Conclusion** — does the data support or reject the "
            "hypothesis?\n"
            "8. **Communication** — publish so others can replicate.\n\n"
            "Key principle: a hypothesis must be falsifiable — there must be "
            "some result that could prove it wrong."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between qualitative and quantitative research?",
        "input": "",
        "output": (
            "**Quantitative research** collects numerical data and uses "
            "statistical analysis. Aims to measure, compare, and generalise.\n"
            "Examples: surveys with Likert scales, experiments, census data.\n"
            "Good for: testing hypotheses, finding patterns across large "
            "samples.\n\n"
            "**Qualitative research** collects non-numerical data — words, "
            "observations, themes. Aims to understand meaning and experience.\n"
            "Examples: interviews, focus groups, case studies, ethnography.\n"
            "Good for: exploring complex topics in depth, generating "
            "hypotheses.\n\n"
            "**Mixed methods** combines both — e.g. a survey (quantitative) "
            "followed by interviews (qualitative) to explain the numbers.\n\n"
            "Neither is superior — the choice depends on your research "
            "question."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between validity and reliability in research?",
        "input": "",
        "output": (
            "**Reliability** — consistency. Does the measure give the same "
            "result when repeated under the same conditions?\n"
            "Example: a bathroom scale that always reads 2 kg too high is "
            "reliable (consistent) but not valid.\n\n"
            "**Validity** — accuracy. Does the measure actually capture what "
            "it claims to measure?\n"
            "Example: asking 'how many books do you own?' to measure "
            "intelligence is reliable but not a valid measure of intelligence."
            "\n\n"
            "Types of validity:\n"
            "• **Internal** — the study design truly isolates cause and effect."
            "\n"
            "• **External** — findings generalise beyond the study sample.\n"
            "• **Construct** — the measure reflects the theoretical concept.\n"
            "\n"
            "Aim for both, but a study can be reliable without being valid."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is linear regression?",
        "input": "",
        "output": (
            "Linear regression models the relationship between one or more "
            "input variables (features) and a continuous output variable by "
            "fitting a straight line.\n\n"
            "Simple linear regression (one feature):\n"
            "  y = β₀ + β₁x + ε\n"
            "  β₀ = intercept, β₁ = slope, ε = error term.\n\n"
            "The model is fitted by minimising the sum of squared residuals "
            "(ordinary least squares).\n\n"
            "Assumptions:\n"
            "• Linear relationship between x and y.\n"
            "• Residuals are normally distributed with constant variance.\n"
            "• No multicollinearity (for multiple regression).\n\n"
            "Evaluation: R² (proportion of variance explained); RMSE (average "
            "prediction error in original units).\n\n"
            "Multiple regression extends this to many features: "
            "y = β₀ + β₁x₁ + β₂x₂ + …"
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the first law of thermodynamics?",
        "input": "",
        "output": (
            "The first law of thermodynamics states that energy cannot be "
            "created or destroyed — only converted from one form to another.\n"
            "\n"
            "Formal statement: the change in internal energy of a system "
            "equals the heat added to the system minus the work done by the "
            "system.\n"
            "  ΔU = Q − W\n"
            "  ΔU = change in internal energy\n"
            "  Q  = heat added to the system\n"
            "  W  = work done by the system\n\n"
            "Examples:\n"
            "• Burning fuel: chemical energy → heat + mechanical work.\n"
            "• A battery: chemical energy → electrical energy.\n"
            "• A fridge: electrical energy moves heat from inside to outside.\n"
            "\n"
            "Implication: a perpetual motion machine of the first kind "
            "(producing energy from nothing) is impossible."
        ),
        "category": "concept_explanation",
    },
    {
        "instruction": "What is the difference between kinetic and potential energy?",
        "input": "",
        "output": (
            "Both are forms of mechanical energy.\n\n"
            "**Kinetic energy (KE)** — energy an object has due to its "
            "motion.\n"
            "  KE = ½mv²  (m = mass in kg, v = velocity in m/s)\n"
            "  Example: a moving car, a thrown ball.\n\n"
            "**Potential energy (PE)** — stored energy due to position or "
            "condition.\n"
            "  Gravitational PE = mgh  (h = height above reference point)\n"
            "  Example: a book on a shelf, a compressed spring.\n\n"
            "Conservation of energy: KE and PE convert into each other. "
            "A ball falling from height h converts PE into KE — at the "
            "bottom all PE has become KE (ignoring air resistance):\n"
            "  mgh = ½mv² → v = √(2gh)"
        ),
        "category": "concept_explanation",
    },

    # ── flashcard_generation — batch 2 ────────────────────────────────────────

    {
        "instruction": "Create 5 flashcards on sorting algorithms.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is the time complexity of bubble sort (worst case)?\n"
            "A: O(n²) — nested loops comparing adjacent elements.\n\n"
            "**Flashcard 2**\n"
            "Q: What is the time complexity of merge sort?\n"
            "A: O(n log n) in all cases — divide and conquer.\n\n"
            "**Flashcard 3**\n"
            "Q: What is quicksort's average time complexity?\n"
            "A: O(n log n); worst case O(n²) with a bad pivot choice.\n\n"
            "**Flashcard 4**\n"
            "Q: Which sort is stable?\n"
            "A: Merge sort is stable (preserves the relative order of equal "
            "elements). Quicksort is typically not.\n\n"
            "**Flashcard 5**\n"
            "Q: When is insertion sort the best choice?\n"
            "A: For small arrays (n < 20) or nearly-sorted data — O(n) best "
            "case, low overhead."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Make 5 flashcards on Git commands.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What does `git clone <url>` do?\n"
            "A: Creates a local copy of a remote repository.\n\n"
            "**Flashcard 2**\n"
            "Q: What does `git status` show?\n"
            "A: Which files are staged, unstaged, or untracked.\n\n"
            "**Flashcard 3**\n"
            "Q: What is `git stash`?\n"
            "A: Temporarily saves uncommitted changes so you can switch "
            "branches without committing.\n\n"
            "**Flashcard 4**\n"
            "Q: What is the difference between `git merge` and `git rebase`?\n"
            "A: Merge preserves the full branch history with a merge commit. "
            "Rebase rewrites commits onto the target branch for a linear "
            "history.\n\n"
            "**Flashcard 5**\n"
            "Q: How do you undo the last commit but keep the changes staged?\n"
            "A: `git reset --soft HEAD~1`"
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards on the SOLID principles.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What does the S in SOLID stand for?\n"
            "A: Single Responsibility Principle — a class should have only "
            "one reason to change.\n\n"
            "**Flashcard 2**\n"
            "Q: What is the Open/Closed Principle?\n"
            "A: Classes should be open for extension but closed for "
            "modification.\n\n"
            "**Flashcard 3**\n"
            "Q: What is the Liskov Substitution Principle?\n"
            "A: Objects of a subclass should be usable wherever objects of "
            "the parent class are expected, without breaking the programme.\n\n"
            "**Flashcard 4**\n"
            "Q: What is the Interface Segregation Principle?\n"
            "A: Clients should not be forced to depend on interfaces they do "
            "not use — prefer many small interfaces over one large one.\n\n"
            "**Flashcard 5**\n"
            "Q: What is the Dependency Inversion Principle?\n"
            "A: High-level modules should not depend on low-level modules; "
            "both should depend on abstractions."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Generate 5 flashcards on calculus differentiation rules.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is the power rule?\n"
            "A: d/dx [xⁿ] = nxⁿ⁻¹\n\n"
            "**Flashcard 2**\n"
            "Q: What is the product rule?\n"
            "A: d/dx [uv] = u'v + uv'\n\n"
            "**Flashcard 3**\n"
            "Q: What is the quotient rule?\n"
            "A: d/dx [u/v] = (u'v − uv') / v²\n\n"
            "**Flashcard 4**\n"
            "Q: What is the chain rule?\n"
            "A: d/dx [f(g(x))] = f'(g(x)) · g'(x)\n\n"
            "**Flashcard 5**\n"
            "Q: What is the derivative of eˣ?\n"
            "A: d/dx [eˣ] = eˣ — it is its own derivative."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards on common logical fallacies.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is an ad hominem fallacy?\n"
            "A: Attacking the person making the argument rather than the "
            "argument itself.\n\n"
            "**Flashcard 2**\n"
            "Q: What is a straw man fallacy?\n"
            "A: Misrepresenting someone's argument to make it easier to "
            "attack.\n\n"
            "**Flashcard 3**\n"
            "Q: What is a false dichotomy?\n"
            "A: Presenting only two options as if they are the only "
            "possibilities, when more exist.\n\n"
            "**Flashcard 4**\n"
            "Q: What is an appeal to authority?\n"
            "A: Claiming something is true simply because an authority figure "
            "says so, without other evidence.\n\n"
            "**Flashcard 5**\n"
            "Q: What is circular reasoning (begging the question)?\n"
            "A: Using the conclusion as a premise in your own argument."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Make 5 flashcards on cognitive biases.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is confirmation bias?\n"
            "A: The tendency to seek, interpret, and remember information "
            "that confirms your existing beliefs.\n\n"
            "**Flashcard 2**\n"
            "Q: What is the availability heuristic?\n"
            "A: Judging how likely something is based on how easily an "
            "example comes to mind.\n\n"
            "**Flashcard 3**\n"
            "Q: What is the sunk cost fallacy?\n"
            "A: Continuing a course of action because of past investment "
            "(time, money) rather than future value.\n\n"
            "**Flashcard 4**\n"
            "Q: What is the Dunning-Kruger effect?\n"
            "A: People with limited knowledge in a domain overestimate their "
            "competence; experts underestimate theirs.\n\n"
            "**Flashcard 5**\n"
            "Q: What is anchoring bias?\n"
            "A: Over-relying on the first piece of information encountered "
            "when making decisions."
        ),
        "category": "flashcard_generation",
    },
    {
        "instruction": "Create 5 flashcards on types of statistical distributions.",
        "input": "",
        "output": (
            "**Flashcard 1**\n"
            "Q: What is a normal distribution?\n"
            "A: A symmetric, bell-shaped distribution defined by its mean "
            "(μ) and standard deviation (σ). About 68 % of values fall "
            "within 1σ of the mean.\n\n"
            "**Flashcard 2**\n"
            "Q: What is a binomial distribution?\n"
            "A: Models the number of successes in n independent trials, each "
            "with probability p.\n\n"
            "**Flashcard 3**\n"
            "Q: What is a Poisson distribution?\n"
            "A: Models the number of events occurring in a fixed interval of "
            "time or space, given a known average rate.\n\n"
            "**Flashcard 4**\n"
            "Q: What is a uniform distribution?\n"
            "A: Every outcome has equal probability (e.g. rolling a fair die)."
            "\n\n"
            "**Flashcard 5**\n"
            "Q: When do you use a t-distribution instead of a normal?\n"
            "A: When the sample size is small (n < 30) and the population "
            "standard deviation is unknown."
        ),
        "category": "flashcard_generation",
    },

    # ── practice_question — batch 2 ───────────────────────────────────────────

    {
        "instruction": "Give me 3 practice problems on calculus derivatives with solutions.",
        "input": "",
        "output": (
            "**Problem 1** — Power rule\n"
            "Find f'(x) for f(x) = 5x³ − 2x + 7.\n"
            "Solution: f'(x) = 15x² − 2\n\n"
            "**Problem 2** — Chain rule\n"
            "Find dy/dx for y = (3x + 1)⁴.\n"
            "Solution: Let u = 3x + 1. dy/dx = 4u³ · 3 = 12(3x+1)³\n\n"
            "**Problem 3** — Product rule\n"
            "Find d/dx [x² · sin x].\n"
            "Solution: u = x², v = sin x\n"
            "  u' = 2x, v' = cos x\n"
            "  d/dx = u'v + uv' = 2x sin x + x² cos x"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice questions on electric circuits with solutions.",
        "input": "",
        "output": (
            "**Problem 1** — Series circuit\n"
            "Three resistors of 2 Ω, 3 Ω, and 5 Ω are in series with a 20 V "
            "battery. Find the current.\n"
            "Solution: R_total = 2+3+5 = 10 Ω. I = V/R = 20/10 = 2 A\n\n"
            "**Problem 2** — Parallel circuit\n"
            "Two resistors, 4 Ω and 6 Ω, are in parallel. Find equivalent "
            "resistance.\n"
            "Solution: 1/R = 1/4 + 1/6 = 3/12 + 2/12 = 5/12 → R = 2.4 Ω\n\n"
            "**Problem 3** — Power\n"
            "A 60 W bulb runs on 120 V. Find the current and resistance.\n"
            "Solution: I = P/V = 60/120 = 0.5 A. R = V/I = 120/0.5 = 240 Ω"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 genetics practice questions using Punnett squares.",
        "input": "",
        "output": (
            "**Question 1**\n"
            "Two heterozygous parents (Tt × Tt) — what is the probability of "
            "a tall offspring (T dominant)?\n"
            "Punnett square: TT, Tt, Tt, tt → 3 tall : 1 short → 75 %\n\n"
            "**Question 2**\n"
            "A homozygous dominant (RR) parent crosses with a homozygous "
            "recessive (rr). What are the offspring genotypes?\n"
            "All offspring: Rr (heterozygous) — 100 % dominant phenotype.\n\n"
            "**Question 3**\n"
            "Two carriers of a recessive disease (Aa × Aa). What fraction of "
            "offspring will have the disease (aa)?\n"
            "Punnett: AA, Aa, Aa, aa → 1/4 = 25 % will be affected."
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice problems on compound interest with solutions.",
        "input": "",
        "output": (
            "**Problem 1**\n"
            "£2,000 invested at 5 % annual interest, compounded yearly, for "
            "3 years. Find the final amount.\n"
            "Formula: A = P(1 + r)ⁿ = 2000(1.05)³ = 2000 × 1.1576 = £2,315.25"
            "\n\n"
            "**Problem 2**\n"
            "Same as above but compounded monthly.\n"
            "A = 2000(1 + 0.05/12)^(12×3) = 2000(1.004167)^36 ≈ £2,321.03\n\n"
            "**Problem 3**\n"
            "How long to double £1,000 at 7 % compounded annually? "
            "(Rule of 72)\n"
            "Approximate: 72 / 7 ≈ 10.3 years.\n"
            "Exact: n = ln(2)/ln(1.07) = 0.693/0.0677 ≈ 10.24 years."
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 Python practice problems on list comprehensions with solutions.",
        "input": "",
        "output": (
            "**Problem 1** — Squares of even numbers\n"
            "Create a list of squares of even numbers from 0 to 19.\n"
            "```python\n"
            "result = [x**2 for x in range(20) if x % 2 == 0]\n"
            "# [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]\n"
            "```\n\n"
            "**Problem 2** — Flatten a 2D list\n"
            "Flatten [[1,2],[3,4],[5,6]] into a single list.\n"
            "```python\n"
            "flat = [x for row in [[1,2],[3,4],[5,6]] for x in row]\n"
            "# [1, 2, 3, 4, 5, 6]\n"
            "```\n\n"
            "**Problem 3** — Word lengths\n"
            "Given a sentence, return a dict of word → length.\n"
            "```python\n"
            "sentence = 'the quick brown fox'\n"
            "lengths = {word: len(word) for word in sentence.split()}\n"
            "# {'the': 3, 'quick': 5, 'brown': 5, 'fox': 3}\n"
            "```"
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice problems on time complexity analysis.",
        "input": "",
        "output": (
            "**Problem 1**\n"
            "What is the time complexity of this code?\n"
            "```python\n"
            "for i in range(n):\n"
            "    for j in range(n):\n"
            "        print(i, j)\n"
            "```\n"
            "Answer: O(n²) — two nested loops each running n times.\n\n"
            "**Problem 2**\n"
            "What is the complexity of looking up a key in a Python dict?\n"
            "Answer: O(1) average case — dicts are backed by hash tables.\n\n"
            "**Problem 3**\n"
            "You sort a list of n items (O(n log n)) then do a binary search "
            "(O(log n)). What is the overall complexity?\n"
            "Answer: O(n log n) — dominated by the sort step. When combining "
            "sequential operations, take the slowest term."
        ),
        "category": "practice_question",
    },
    {
        "instruction": "Give me 3 practice questions on balancing chemical equations.",
        "input": "",
        "output": (
            "**Problem 1** — Combustion of methane\n"
            "Balance: CH₄ + O₂ → CO₂ + H₂O\n"
            "Balanced: CH₄ + 2O₂ → CO₂ + 2H₂O\n"
            "(4 H on left → 2 H₂O; 2+1=2 O on right → 2 O₂)\n\n"
            "**Problem 2** — Synthesis of water\n"
            "Balance: H₂ + O₂ → H₂O\n"
            "Balanced: 2H₂ + O₂ → 2H₂O\n\n"
            "**Problem 3** — Iron + oxygen → iron oxide\n"
            "Balance: Fe + O₂ → Fe₂O₃\n"
            "Balanced: 4Fe + 3O₂ → 2Fe₂O₃\n"
            "(4 Fe on each side; 6 O on each side ✓)"
        ),
        "category": "practice_question",
    },

    # ── study_skills — batch 2 ────────────────────────────────────────────────

    {
        "instruction": "How do I read an academic paper effectively?",
        "input": "",
        "output": (
            "Academic papers are not meant to be read cover to cover on the "
            "first pass. Use a structured approach.\n\n"
            "**Pass 1 — Skim (5 min)**\n"
            "Read: title, abstract, introduction, headings, conclusion, "
            "figures. Decide: is this paper relevant and worth deeper reading?"
            "\n\n"
            "**Pass 2 — Read carefully (30–60 min)**\n"
            "Read everything, but skip the maths/proofs for now. Note the "
            "research question, methodology, key findings, and limitations.\n\n"
            "**Pass 3 — Deep dive (if needed)**\n"
            "Work through equations, re-read unclear sections, check cited "
            "sources.\n\n"
            "While reading, note:\n"
            "• What problem does this solve?\n"
            "• What is their evidence?\n"
            "• What do they claim NOT to cover?\n"
            "• Does this support or challenge what I already know?"
        ),
        "category": "study_skills",
    },
    {
        "instruction": "How do I stay motivated when studying online?",
        "input": "",
        "output": (
            "Online study removes the natural structure of a classroom, so "
            "motivation has to be built deliberately.\n\n"
            "Strategies that work:\n\n"
            "**Set a clear schedule** — treat online sessions like classes "
            "you cannot skip. Block the time in your calendar.\n\n"
            "**Define small daily goals** — 'finish chapter 3 exercises' "
            "is more motivating than 'study maths'.\n\n"
            "**Create a study environment** — a dedicated desk with no phone "
            "signals your brain it's time to work.\n\n"
            "**Track progress visibly** — a checklist or streak counter gives "
            "a dopamine hit from completion.\n\n"
            "**Study with others** — virtual co-working sessions (on Discord "
            "or Zoom) recreate the social accountability of a library.\n\n"
            "**Reward yourself** — plan something enjoyable after hitting "
            "your daily target."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "What is the SQ3R reading method?",
        "input": "",
        "output": (
            "SQ3R is a structured reading strategy that improves comprehension "
            "and retention of textbook material.\n\n"
            "**S — Survey**: Skim the chapter first. Read headings, bold "
            "terms, intro, and summary. Build a mental map.\n\n"
            "**Q — Question**: Turn each heading into a question before "
            "reading that section. "
            "E.g. 'Types of Memory' → 'What are the types of memory?'\n\n"
            "**R — Read**: Read actively, looking for the answer to your "
            "question.\n\n"
            "**R — Recite**: After each section, close the book and answer "
            "your question aloud or in writing.\n\n"
            "**R — Review**: At the end, go back through your notes and quiz "
            "yourself on the whole chapter.\n\n"
            "Why it works: it forces active engagement instead of passive "
            "highlighting."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "How do I use mind maps for studying?",
        "input": "",
        "output": (
            "A mind map is a visual diagram that organises information around "
            "a central topic, showing how ideas connect.\n\n"
            "How to make one:\n"
            "1. Write the main topic in the centre and circle it.\n"
            "2. Draw branches for major sub-topics.\n"
            "3. Add smaller branches for details, examples, or definitions.\n"
            "4. Use colours, icons, and short phrases — not full sentences.\n"
            "\n"
            "Best uses for study:\n"
            "• Summarising a chapter after reading.\n"
            "• Planning an essay structure.\n"
            "• Revising a large topic before an exam.\n"
            "• Seeing how different concepts link together.\n\n"
            "Tools: pen and paper is best for memory. Digitally: XMind, "
            "Miro, or Obsidian.\n\n"
            "Tip: draw the map from memory first, then check your notes and "
            "fill in gaps — this is active recall."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "How do I manage a group project effectively at university?",
        "input": "",
        "output": (
            "Group projects fail most often from poor communication and "
            "unclear responsibilities — not lack of effort.\n\n"
            "First meeting:\n"
            "• Read the brief together and agree on the goal.\n"
            "• Split work by skill or preference, not randomly.\n"
            "• Set individual deadlines before the group deadline "
            "(buffer time matters).\n"
            "• Agree on a communication channel (WhatsApp, Slack, etc.).\n\n"
            "Throughout the project:\n"
            "• Hold short weekly check-ins — 15 min is enough.\n"
            "• If someone is behind, flag it early — not the night before.\n"
            "• Keep a shared document for progress tracking.\n\n"
            "Final stage:\n"
            "• Integrate everyone's sections early — late integration always "
            "reveals inconsistencies.\n"
            "• One person should proofread the final version for coherence."
        ),
        "category": "study_skills",
    },
    {
        "instruction": "What digital tools are most useful for university students?",
        "input": "",
        "output": (
            "Here are high-value tools organised by task:\n\n"
            "**Note-taking**\n"
            "• Notion — flexible workspace for notes, tasks, databases.\n"
            "• Obsidian — local markdown notes with linked graphs.\n"
            "• OneNote — good for handwriting and diagrams.\n\n"
            "**Flashcards / memorisation**\n"
            "• Anki — spaced repetition flashcards. Best for languages, "
            "medicine, and anything that needs memorising.\n\n"
            "**Focus**\n"
            "• Forest / Focusmate — Pomodoro and accountability.\n"
            "• Freedom / Cold Turkey — block distracting sites.\n\n"
            "**Research**\n"
            "• Zotero — free reference manager. Generates citations "
            "automatically.\n"
            "• Google Scholar — academic paper search.\n\n"
            "**Writing**\n"
            "• Grammarly — grammar and clarity.\n"
            "• Hemingway Editor — flags overly complex sentences.\n\n"
            "Start with one tool per category — too many tools becomes "
            "procrastination."
        ),
        "category": "study_skills",
    },

    # ── exam_prep — batch 2 ───────────────────────────────────────────────────

    {
        "instruction": "How should I prepare for and tackle a maths or statistics exam?",
        "input": "",
        "output": (
            "Maths exams are won through practice, not re-reading.\n\n"
            "**Preparation**\n"
            "• Do problems, not passive review — close the textbook and "
            "attempt questions.\n"
            "• Work through past papers under timed conditions.\n"
            "• Write out all key formulas on a single sheet; memorise them.\n"
            "• Understand derivations, not just results — examiners often "
            "ask 'show that'.\n\n"
            "**In the exam**\n"
            "• Show all working — partial marks are awarded even for wrong "
            "answers.\n"
            "• Check units and dimensional analysis.\n"
            "• If stuck, write what you know and move on — return later.\n"
            "• Estimate the answer before calculating; it helps catch "
            "obvious errors.\n"
            "• Leave 5 minutes to re-check arithmetic."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How do I approach an open-book exam?",
        "input": "",
        "output": (
            "Open-book exams test application and analysis, not memorisation "
            "— so over-reliance on your notes is a trap.\n\n"
            "Before the exam:\n"
            "• Organise your notes and textbook with tabs or bookmarks so "
            "you can find things quickly.\n"
            "• Know your material well enough that you only need notes "
            "for specific details (formulas, dates, quotes).\n"
            "• Create a one-page summary of the most important concepts.\n\n"
            "During the exam:\n"
            "• Don't spend half your time searching through notes — if it "
            "takes more than 1 minute to find something, answer from memory "
            "and move on.\n"
            "• Focus on answering the question, not copying from the book.\n"
            "• Cite sources where required but don't quote-dump."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How do I write a strong introduction for an exam essay?",
        "input": "",
        "output": (
            "A strong exam essay introduction does three things in 3–5 "
            "sentences:\n\n"
            "1. **Contextualise** — briefly set up the topic (one sentence).\n"
            "2. **Thesis** — state your direct answer to the question. "
            "Don't be vague — examiners reward clarity.\n"
            "3. **Signpost** — outline the main points you will cover "
            "('This essay will argue… by examining…').\n\n"
            "Example structure:\n"
            "'Climate change is one of the defining challenges of the 21st "
            "century. This essay argues that carbon pricing is the most "
            "effective policy response because it aligns economic incentives "
            "with environmental goals. It will first examine the evidence for "
            "carbon taxes, then compare alternatives, before concluding with "
            "a policy recommendation.'\n\n"
            "Avoid: starting with a dictionary definition or vague "
            "statements like 'This is a complex topic.'"
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How should I answer a data interpretation question in an exam?",
        "input": "",
        "output": (
            "Data interpretation questions give you a table, graph, or chart "
            "and ask you to draw conclusions.\n\n"
            "Step-by-step approach:\n"
            "1. **Read the title and axes first** — understand what is being "
            "measured and in what units.\n"
            "2. **Identify the trend** — is the relationship positive, "
            "negative, flat, cyclical?\n"
            "3. **Spot anomalies** — are there outliers or unexpected dips?\n"
            "4. **Quantify when possible** — 'increased by 40 %' is stronger "
            "than 'went up'.\n"
            "5. **Relate back to the question** — don't describe everything, "
            "only what is relevant.\n"
            "6. **Avoid over-claiming** — a graph showing correlation does "
            "not prove causation.\n\n"
            "Common mistake: describing what you see without interpreting "
            "what it means."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How do I write a lab report?",
        "input": "",
        "output": (
            "A lab report documents an experiment so others can evaluate and "
            "reproduce it. Standard structure:\n\n"
            "**Title** — descriptive, specific.\n\n"
            "**Abstract** (100–200 words) — aim, method summary, key result, "
            "conclusion.\n\n"
            "**Introduction** — background, theory, research question, "
            "hypothesis.\n\n"
            "**Methods** — step-by-step procedure in past tense, passive "
            "voice. Include equipment, quantities, controls.\n\n"
            "**Results** — present data as tables/graphs. Label everything. "
            "Describe trends, don't interpret yet.\n\n"
            "**Discussion** — interpret results. Do they support the "
            "hypothesis? How do they compare with literature? What are "
            "sources of error?\n\n"
            "**Conclusion** — one short paragraph: what was found and "
            "its significance.\n\n"
            "**References** — cite any sources used."
        ),
        "category": "exam_prep",
    },
    {
        "instruction": "How do I handle a question I haven't seen before in an exam?",
        "input": "",
        "output": (
            "Unseen questions test whether you can apply knowledge, not "
            "just recall it. They are intentional.\n\n"
            "What to do:\n"
            "1. **Read it twice** — identify exactly what is being asked. "
            "Underline the command word (analyse, compare, evaluate).\n"
            "2. **Connect to what you know** — what topic does this relate "
            "to? What principles apply?\n"
            "3. **Plan before writing** — 1–2 minutes sketching an outline "
            "prevents rambling.\n"
            "4. **Define your terms** — starting with definitions shows "
            "understanding and buys thinking time.\n"
            "5. **Reason out loud on paper** — show your working, even if "
            "you're unsure. Examiners credit method.\n"
            "6. **Don't leave it blank** — a partial answer always beats "
            "nothing.\n\n"
            "Mindset: the question isn't designed to trick you — it's "
            "designed to test reasoning."
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
