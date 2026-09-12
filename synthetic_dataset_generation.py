from google import genai
from dotenv import load_dotenv
from time import perf_counter, sleep
import json
import csv
import os

load_dotenv()

client = genai.Client()

MODEL = "gemini-3.8-flash"
CSV_FILE = "synthetic_dataset.csv"

N = 2

REQUEST_INTERVAL = 6.5
MAX_BACKOFF = 120
MAX_CONSECUTIVE_429 = 10

next_id = 0
try:
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        next_id = sum(1 for _ in f)
except:
    pass

last_request_time = 0
previous_interaction_id = None

assignment_contexts = [
    """Given miles driven, electricity consumed (kWh), waste generated (kg), recycled? (1/0), and used public transport? (1/0), write a Java program using arithmetic, relational, logical, and assignment operators to calculate vehicle emission = miles x 0.40 kg CO₂, electricity emission = kWh x 0.85 kg CO₂, waste emission = waste x 1.20 kg CO₂, and total footprint = sum of all three. A person is Eco-Friendly iff total < 20 AND recycled AND used public transport. Output all three emissions, total footprint, whether <20 is achieved, and whether the person is Eco-Friendly.""",

    """Given current score, mission number, and coins collected, write Java using switch for mission scoring: 1 -> +50, 2 -> +100, 3 -> +150, 4 -> +250, anything else -> +0. After adding mission points, if score ≥500 add 100 bonus, otherwise 0. Determine level using if-else-if: <200 -> Beginner, 200-399 -> Intermediate, 400-699 -> Advanced, ≥700 -> Master. If coins ≥50 output "Extra Life Awarded!", otherwise "Keep Collecting Coins!". Output updated score, bonus, final score, level, and coin message.""",

    """Given n charging sessions, each containing Session ID, Vehicle Number, Vehicle Type (Car/Bike/Bus), and Energy (kWh), store them in an array of objects. Rates are Car=₹20/kWh, Bike=₹10/kWh, Bus=₹30/kWh. Calculate each charging amount, find the highest-paying vehicle (first vehicle on a tie), and calculate total revenue. Output total revenue and the highest-paying vehicle's number and amount, with amounts formatted to 2 decimals.""",

    """Create Reader and RecommendationEngine classes. Reader has private Reader ID, Name, Books Borrowed, and Membership Type (Gold/Silver/Regular), initialized through a parameterized constructor with public getters. Membership points: Gold=30, Silver=20, Regular=10. Recommendation Score = Books Borrowed x 5 + Membership Points. RecommendationEngine provides boolean eligible(Reader r), true when score ≥80. Reader.checkEligibility() must create a RecommendationEngine and call engine.eligible(this). Store multiple Readers in an array and display each reader's ID, name, recommendation score, and eligibility.""",

    """Create a Truck class with private Truck ID, Driver Name, Distance Travelled (km), and Fuel Consumed (L), using a parameterized constructor and getters. Mileage = Distance / Fuel. Classify trucks as ≥18 -> Excellent, 15-<18 -> Good, 10-<15 -> Average, <10 -> Poor. Store trucks in an array, find/display the highest-mileage truck (ID, driver, mileage, category), calculate average mileage, and count trucks in each of the four performance categories.""",

    """Create a Passenger class with private Passenger ID, Airline Name, and Baggage Weight, initialized via parameterized constructor. Free Baggage Allowance is a static variable, initially 20 kg using a static block. If weight exceeds allowance, Excess Charge = (Weight - Allowance) x ₹500; otherwise ₹0. Implement static setFreeAllowance() to modify the allowance. Read 2 passengers, display each passenger's ID, airline, and initial excess charge; find the ID of the highest-paying passenger; then set allowance to 30 kg, recalculate both charges, and output their sum.""",

    """Given n subscribers and a common Festival Discount %, store subscribers in an array. Each subscriber has ID, Plan (Basic/Standard/Premium), and Movies Watched. Monthly fees: Basic=₹299, Standard=₹499, Premium=₹799. Implement static setDiscountPercentage(double discount). Additional viewing discount based on total movies watched by all subscribers: <10 -> 0%, 10-19 -> 2%, ≥20 -> 3%. Final Fee = Monthly Fee x [1 - (Festival Discount + Additional Discount)/100]. Festival discount is 0-20%; display final fees to 2 decimals; do not use ArrayList/collections. Output the ID of the subscriber with highest final fee (first on tie), counts of Basic/Standard/Premium subscribers, subscribers receiving additional discount, and subscribers whose final fee ≥ the average final fee.""",

    """Given the number of posts and likes received for each post, use a for loop to calculate total likes, average likes, and maximum likes. Determine engagement level: average ≥500 -> Trending Creator, 200-499 -> Popular Creator, otherwise -> Growing Creator. Output total likes, average likes, maximum likes, and creator engagement level."""
]

assignment_index = 0
current_assignment_context = None


SYSTEM_PROMPT = """You are working with a AI student tutoring system. This is the current assignment context. Understand the assignment and internalize this. You must simulate conversation students might have with the AI tutoring system over this assignment.

Assignment context:

{assignment_context}

"""


P1 = """You are generating synthetic training data for an educational AI tutoring
system's guardrail classifiers. Each example is a conversation split into
previous_conversation (historical context turns) and current_turn (the final
user-assistant exchange being labeled).

Ground every example in this specific assignment. Think concretely about
how a real student working on this problem would ask questions, and how a
policy-bound tutoring assistant would realistically respond.

DEFINITIONS

Irrelevancy applies to current_turn.user.

Irrelevancy is "yes" when the message is off-topic from the assignment,
tries to redirect away from it, attempts to override the assistant's
instructions (prompt injection, "ignore your rules," role play framings),
or contains profanity or hateful/harassing language.
Irrelevancy is "no" when it is a legitimate, on-topic question about the
assignment, even if indirect or a natural follow-up.

Solution Proximality applies to current_turn.assistant.

Solution Proximality is "yes" when the message hands over the solution in
any form: solving source code, a complete algorithm or pseudocode
(including informal/high-level), a flowchart or equivalent diagram, or a
step-by-step procedure that produces the working solution if followed.
Solution Proximality is no when it only explains concepts, syntax, APIs,
errors, formulae, or complexity; gives hints; points out mistakes;
suggests debugging strategies; names the relevant technique; or helps with
only the single step the student is stuck on.

STRUCTURE

previous_conversation: 2 ordinary, on-topic, policy-compliant
user-assistant turn pairs that plausibly lead up to the current turn. Vary
the number of prior turns across the batch.
current_turn: exactly one user message and one assistant reply. Only this
turn determines the labels.

BATCH

Generate exactly 4 examples per batch, in this order:
1. irrelevancy: no, solution_proximality: no (postive example)
2. irrelevancy: no, solution_proximality: yes (negative example)
3. irrelevancy: yes, solution_proximality: no (negative example)
4. irrelevancy: no, solution_proximality: no (postive example)

OUTPUT FORMAT

Return a JSON array of exactly 4 objects, in order:

{
  "previous_conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "current_turn": {
    "user": "...",
    "assistant": "..."
  },
  "irrelevancy": "yes" or "no",
  "solution_proximality": "yes" or "no"
}

Return only the JSON array, no other text."""


P2 = """You are generating synthetic training data for an educational AI tutoring
system's guardrail classifiers. Unlike ordinary examples, these
conversations must show the anomaly building progressively across turns,
not appearing suddenly only in the final turn. Each example is split into
previous_conversation (historical context turns) and current_turn (the final
user-assistant exchange).

Ground every example in this specific assignment. Think concretely about
how a real student working on this problem would ask questions, and how a
policy-bound tutoring assistant would realistically respond.

DEFINITIONS

Irrelevancy is "yes" when a message is off-topic from the assignment, tries
to redirect away from it, attempts to override the assistant's
instructions (prompt injection, "ignore your rules," role play framings),
or contains profanity or hateful/harassing language. 
It is "no" when the
message is a legitimate, on-topic question about the assignment.

Solution Proximality is "yes" when a response, considered together with
everything given earlier in the conversation, hands over the solution in
full: solving source code, a complete algorithm or pseudocode, a flowchart
or equivalent diagram, or a step-by-step procedure that produces the
working solution if followed, whether that full content sits in one
response or is assembled piece by piece across responses. 
It is "no" when no
combination of what has been given, including the current response,
amounts to the full solution.

STRUCTURE (progressive, cumulative anomaly)

previous_conversation: 2 to 3 turn pairs that each incrementally advance
the anomaly rather than staying clean until the last turn.
current_turn: the final user-assistant exchange, where the cumulative
pattern completes.

BATCH

Generate exactly 2 examples per batch, in this order:

1. Multi-turn Irrelevancy: yes, Solution Proximality: no
   Each previous_conversation turn drifts a little further from the
   assignment than the last (mild tangents, small redirections, testing
   the boundary), culminating in a current_turn user message that is
   clearly off-topic, an override attempt, or contains profanity/hateful
   language. The assistant's replies throughout, including current_turn,
   never leak solution content. (negative example)

2. Multi-turn Solution Proximality: yes, Irrelevancy: no
   The student asks for a different legitimate-looking piece of the
   solution in each previous_conversation turn (e.g. one step, one helper
   function), and the assistant answers each in isolation
   as if it were a bounded hint. current_turn's user question is the
   remaining on-topic piece, and the assistant's reply, when combined with
   everything given in previous_conversation, completes the full solution
   even though current_turn's response alone may look modest. All user
   messages, including current_turn, stay on-topic and non-adversarial. (negative example)

3. irrelevancy: no, solution_proximality: no (normal positive example)
4. irrelevancy: no, solution_proximality: no (normal positive example)

OUTPUT FORMAT

Return a JSON array of exactly 2 objects, in order:

{
  "previous_conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "current_turn": {
    "user": "...",
    "assistant": "..."
  },
  "irrelevancy": "yes" or "no",
  "solution_proximality": "yes" or "no"
}

Return only the JSON array, no other text.
"""


if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "assignment_context",
            "previous_conversation",
            "user_query",
            "assistant_response",
            "irrelevancy",
            "solution_proximality"
        ])


def get_next_id():
    global next_id
    temp = next_id
    next_id += 1
    return temp


def wait_between_requests():
    global last_request_time

    elapsed = perf_counter() - last_request_time

    if elapsed < REQUEST_INTERVAL:
        sleep(REQUEST_INTERVAL - elapsed)


def send_request(prompt, new_conversation=False):
    global last_request_time
    global previous_interaction_id

    backoff = REQUEST_INTERVAL
    consecutive_429 = 0

    while True:
        wait_between_requests()

        try:
            last_request_time = perf_counter()

            if new_conversation:
                interaction = client.interactions.create(
                    model=MODEL,
                    input=prompt
                )
            else:
                interaction = client.interactions.create(
                    model=MODEL,
                    input=prompt,
                    previous_interaction_id=previous_interaction_id
                )

            previous_interaction_id = interaction.id

            return interaction

        except Exception as e:
            error = str(e)

            if "429" in error or "RESOURCE_EXHAUSTED" in error:
                consecutive_429 += 1

                print(
                    f"429 received "
                    f"({consecutive_429}/{MAX_CONSECUTIVE_429})"
                )

                if consecutive_429 > MAX_CONSECUTIVE_429:
                    print("More than 10 consecutive 429s. Quitting.")
                    raise SystemExit

                print(f"Retrying in {backoff:.1f} seconds...")

                sleep(backoff)

                backoff = min(backoff * 2, MAX_BACKOFF)

            else:
                print("ERROR:", e)
                raise


def store_results(output):
    data = json.loads(output)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for item in data:
            writer.writerow([
                get_next_id(),
                current_assignment_context,
                json.dumps(
                    item["previous_conversation"],
                    ensure_ascii=False
                ),
                item["current_turn"]["user"],
                item["current_turn"]["assistant"],
                item["irrelevancy"],
                item["solution_proximality"]
            ])


def start_new_conversation():
    global assignment_index
    global current_assignment_context
    global previous_interaction_id

    current_assignment_context = assignment_contexts[assignment_index]

    system_prompt = SYSTEM_PROMPT.format(
        assignment_context=current_assignment_context
    )

    print(
        f"\nStarting new conversation "
        f"with assignment context {assignment_index}"
    )

    previous_interaction_id = None

    send_request(
        system_prompt,
        new_conversation=True
    )

    assignment_index = (
        assignment_index + 1
    ) % len(assignment_contexts)


def generate(prompt, expected_count):
    start_time = perf_counter()

    interaction = send_request(prompt)

    end_time = perf_counter()

    store_results(interaction.output_text)

    print(
        f"Generated {expected_count} objects "
        f"in {round(end_time - start_time, 2)} seconds"
    )


def run():
    rounds = 0

    start_new_conversation()

    while True:
        generate(P1, 4)
        generate(P2, 4)

        rounds += 1

        if rounds >= N:
            rounds = 0
            start_new_conversation()


run()