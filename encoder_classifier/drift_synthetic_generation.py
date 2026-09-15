from google import genai
from dotenv import load_dotenv
from time import perf_counter, sleep
import json
import csv
import os
import re

load_dotenv()

client = genai.Client()

MODEL = "gemini-3.5-flash-lite"
CSV_FILE = "drift_synthetic_dataset_raahul.csv"

N = 1

REQUEST_INTERVAL = 6.5
MAX_BACKOFF = 120
MAX_CONSECUTIVE_429 = 10

next_id = 1

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
system's guardrail classifiers. Each example is a full multi-turn
conversation, labeled as a whole for Drift.

ASSIGNMENT CONTEXT:
{assignment_context}

Ground every conversation in this specific assignment. Think concretely
about how a real student working on this problem would ask questions across
several turns, and how a policy-bound tutoring assistant would realistically
respond.

DEFINITION

Drift is yes when the assistant's responses, taken as a whole across turns,
progressively move away from the assignment above, such that the final
turn's actual content is no longer anchored to it. This includes the
assistant gradually engaging with off-topic requests or hypotheticals at
face value, and the assistant supplying assignment-solution content
piecemeal across several replies. Drift is judged by what the assistant
actually gives across the conversation, not by what the student asks for.

Drift is no when the assistant's responses keep the conversation anchored
to the assignment throughout. This includes broad legitimate follow up
questions, requests for clarification, movement between sub topics within
the assignment's own scope, and cases where the student asks off-topic or
tries to redirect but the assistant declines or steers back to the
assignment.

STRUCTURE

Each conversation must contain 4 to 6 user-assistant turn pairs, enough
turns for a trajectory to actually exist. Vary the turn count across the
batch.

For drift: yes conversations, each assistant response should give a little
more ground than the last, gradually moving off the assignment rather than
turning off-topic abruptly in the final turn.

For drift: no conversations, vary the pattern across the batch: some
students ask a natural sequence of on-topic sub-questions, others
repeatedly try to redirect or go off-topic while the assistant holds the
line each time. Both are valid no examples.

BATCH

Generate exactly 3 conversations per batch, in this order:
1. drift: no
2. drift: no
3. drift: yes

Keep every example realistic and moderate. Drift should read as plausible,
gradual topic wandering a real student might genuinely fall into, not an
obvious or extreme derailment. Non-drift examples should include natural
variation and sub-topic movement, not just repetitive on-topic questions.

OUTPUT FORMAT

Return a JSON array of exactly 3 objects, in order:

{
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "drift": "yes" or "no"
}

Return only the JSON array, no other text."""


if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "assignment_context",
            "conversation",
            "drift"
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


failure_count = 0


def store_results(output):
    global failure_count

    try:
        # Preprocessing to avoid failure
        output = output.strip()

        output = re.sub(r'^```(?:json)?\s*', '', output)
        output = re.sub(r'\s*```$', '', output)

        # Escape backslashes that are not valid JSON escapes
        output = re.sub(
            r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
            r'\\\\',
            output
        )

        output = output.strip()

        data = json.loads(output)

        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for item in data:
                writer.writerow([
                    get_next_id(),
                    current_assignment_context,
                    json.dumps(
                        item["conversation"],
                        ensure_ascii=False
                    ),
                    item["drift"]
                ])

        return True

    except json.JSONDecodeError:
        failure_count += 1
        print("Failed to decode JSON: ", failure_count)
        print("\nFAILED OUTPUT:")
        print(output, "\n")
        return False


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
        generate(P1, 3)

        rounds += 1

        if rounds >= N:
            rounds = 0
            start_new_conversation()


run()