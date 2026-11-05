"""Restaurant data management.

Reuses the LLM, prompts, schema and repair loop built in:
  - Lesson 1: Extract Structured JSON from Restaurant Text Using LLMs
  - Lesson 2: Process Multimodal Data with LLMs
"""

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

# The generative model from Lesson 1 (local Ollama, not watsonx).
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_predict=512)

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'

# The few-shot example from Lesson 1: restaurant_list[1] and its verified output.
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $'

EXAMPLE_OUTPUT = """
    {
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
    }
"""


class Restaurant(BaseModel):
    """The restaurant pydantic scheme used in lesson 1."""
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


#Update your restaurant_data_structure_prompt_generation
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    you are a restaurent information parser who outputs
    exactly the same structure shown in the Example for any given restaurant description.
    Two rules you have to strictly follow:
    1) It should be very strictly match the structure of the example's output and content from restuarant description.
    2) Never include any tags like json or output . Output should give the raw file as it is.
    """

    base_user_prompt = f"""
    Task:
    give only the raw json for the restuarant in the restuarant description. do not include any tags

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}

    """
    return base_system_msg, base_user_prompt


# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt, params=None):
    # params: the structured-output format passed to the model, e.g. "json".
    # This is the `fmt` argument of the llm_model() defined in Lesson 1.
    messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt_txt)]
    model = llm.bind(format=params) if params else llm
    return model.invoke(messages).content


def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """
    you are a json output error fixing expert who takes the "candidate json output" and
    "error message" for a restaurant data and fixes all the errors and produces the perfect output
    Rules to follow:
    1) Always output the fixed json file.
    2) Output must match the STRUCTURE and SYNTAX of 'validate against' -- same keys, same types.
    3) 'validate against' is a SHAPE EXAMPLE ONLY. Never copy any of its values.
       Every value must come from 'candidate json output' or, if a field is missing there,
       from 'source paragraph'. If a value is genuinely absent, use "" for text,
       null for numbers, and [] for lists.
    4) Do not change values that are already present and correct in 'candidate json output'.
    5) No tags or prose before or after the output, PURE JSON ONLY.
    """
    auto_repair_prompt = f"""
    candidate json output = {response}
    error message = {error_message}
    validate against : {EXAMPLE_OUTPUT}
    """
    return auto_repair_system_msg, auto_repair_prompt


def new_data_entry_process(paragraph, itemId):
    """Structure one new restaurant paragraph into a validated record.

    paragraph: the new restaurant paragraph
    itemId:    the ID to assign to this new item

    Returns the validated record as a Python dict, with 'itemId' attached.
    Raises ValueError if the output cannot be repaired into a valid record.
    """
    ### Step 1: Build the extraction prompts and produce the initial output
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(
        restaurant_paragraph=paragraph
    )
    candidate_json_output = llm_model(
        system_msg=base_system_msg, prompt_txt=base_user_prompt, params="json"
    )

    ### Step 2: Validation and auto-correction loop, same as Lesson 1
    verified = False
    attempts = 0
    error_message = None
    while not verified and attempts < 5:
        try:
            Restaurant.model_validate_json(candidate_json_output)
            verified = True
        except ValidationError as e:
            error_message = e.json()
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(
                candidate_json_output, error_message
            )
            candidate_json_output = llm_model(
                system_msg=auto_repair_system_msg,
                prompt_txt=auto_repair_prompt,
                params="json",
            )
            attempts += 1

    if not verified:
        raise ValueError(
            f"Could not produce a valid restaurant record after {attempts} repair "
            f"attempt(s). Last error: {error_message}"
        )

    ### Step 3: Convert to a dict and attach the itemId, consistent with the review data
    new_entry = json.loads(candidate_json_output)
    new_entry['itemId'] = itemId

    return new_entry


def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data, file_path, backup_path):
    # Create a backup before writing
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    """Displays restaurant data in a clean, vertical format."""
    print(f"\n{'='*15} RESTAURANT #{index} {'='*15}")
    # Prioritize 'name' if it exists, otherwise use 'id'
    name = res.get('name', res.get('restaurant_name', 'Unnamed Restaurant'))
    print(f"NAME        : {name}")

    for key, value in res.items():
        if key.lower() not in ['name', 'restaurant_name']:
            # Handle long descriptions by wrapping slightly
            label = key.replace('_', ' ').upper()
            print(f"{label:<12}: {value}")
    print('='*45)


def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            # Iterate through the records in the data file and show their names.
            # If name doesn't exist, print 'N/A'.
            for index, res in enumerate(data):
                print(f"{index}. {res.get('name', 'N/A')}")

        elif choice == '2':
            # Get the record index in demand from the user, check its validity,
            # and show the card if it is valid.
            index = input("\nRecord index: ")
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = -1
            if 0 <= index < len(data):
                show_restaurant_card(data[index], index)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1 #the item id for the new data

                paragraph = input("\nNew restaurant description: ")
                new_entry = new_data_entry_process(paragraph, itemId)
                data.append(new_entry)
                save_data(data, file_path, backup_path)

                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                index = input("\nRecord index to edit: ")
                try:
                    index = int(index)
                except (TypeError, ValueError):
                    index = -1
                if 0 <= index < len(data):
                    record = data[index]
                    print("Press Enter to keep the current value.")
                    for key, value in list(record.items()):
                        new_value = input(f"{key} [{value}]: ")
                        if new_value == "" or new_value is None:
                            continue
                        # Keep the stored type so the record still validates
                        if isinstance(value, list):
                            new_value = [v.strip() for v in str(new_value).split(",") if v.strip()]
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            new_value = type(value)(new_value)
                        record[key] = new_value
                    data[index] = record
                    save_data(data, file_path, backup_path)
                    print("✅ Record updated.")
                else:
                    print("invalid index.")

            elif choice == '5': # DELETE DATA
                index = input("\nRecord index to delete: ")
                try:
                    index = int(index)
                except (TypeError, ValueError):
                    index = -1
                if 0 <= index < len(data):
                    removed = data.pop(index)
                    save_data(data, file_path, backup_path)
                    print(f"✅ Deleted: {removed.get('name', 'N/A')}")
                else:
                    print("invalid index.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")


class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    # unittest.main() # Unit Test
    manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call
