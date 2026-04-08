"""
Task definitions for the DataQA environment.

Each task provides:
- A clean dataset (CSV)
- A schema + validation rules
- A set of planted issues (ground truth)
- A function to inject those issues into the clean data
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class PlantedIssue:
    """A single planted data quality issue."""

    row: int
    col: str
    issue_type: str
    description: str
    difficulty: float = 1.0  # 1.0=easy, 2.0=medium, 3.0=hard (for weighted reward)

    def to_key(self) -> str:
        return f"row:{self.row},col:{self.col},issue:{self.issue_type}"


@dataclass
class Task:
    task_id: str
    name: str
    description: str
    schema_description: str
    validation_rules: str
    clean_csv: str
    planted_issues: List[PlantedIssue] = field(default_factory=list)
    corrupted_csv: str = ""
    max_steps: int = 3

    def get_clean_value(self, row: int, col: str) -> str | None:
        """
        Look up the original clean value for a given (row, col).
        Row is 1-indexed (data row after header).
        Returns None if row/col is out of bounds or column not found.
        """
        rows = _csv_to_rows(self.clean_csv)
        if len(rows) < 2:
            return None
        header = [h.strip().lower() for h in rows[0]]
        if col.lower() not in header:
            return None
        col_idx = header.index(col.lower())
        data_row_idx = row  # row is 1-indexed, rows[0] is header, so rows[row] is the data row
        if data_row_idx < 1 or data_row_idx >= len(rows):
            return None
        return rows[data_row_idx][col_idx].strip()

    def get_planted_issue_map(self) -> dict:
        """Return dict mapping issue key -> PlantedIssue for quick lookups."""
        return {issue.to_key(): issue for issue in self.planted_issues}


def _csv_to_rows(csv_text: str) -> List[List[str]]:
    reader = csv.reader(io.StringIO(csv_text.strip()))
    return [row for row in reader]


def _rows_to_csv(rows: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()


# ---------------------------------------------------------------------------
# TASK 1: Easy — Employee directory with obvious issues
# ---------------------------------------------------------------------------

def create_task_easy(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = """employee_id,name,email,department,salary,start_date
101,Alice Chen,alice.chen@company.com,Engineering,95000,2022-03-15
102,Bob Martinez,bob.martinez@company.com,Marketing,72000,2021-07-01
103,Carol Davis,carol.davis@company.com,Engineering,98000,2020-11-20
104,David Kim,david.kim@company.com,Sales,68000,2023-01-10
105,Eve Johnson,eve.johnson@company.com,HR,71000,2022-06-05
106,Frank Wilson,frank.wilson@company.com,Engineering,102000,2019-08-12
107,Grace Lee,grace.lee@company.com,Marketing,75000,2021-12-01
108,Hank Brown,hank.brown@company.com,Sales,65000,2023-04-18
109,Iris Patel,iris.patel@company.com,HR,73000,2020-02-28
110,Jack Taylor,jack.taylor@company.com,Engineering,97000,2022-09-14
111,Kevin Zhang,kevin.zhang@company.com,Engineering,91000,2021-05-22
112,Laura Adams,laura.adams@company.com,Sales,69000,2022-11-03
113,Mike Torres,mike.torres@company.com,Marketing,74000,2020-08-17
114,Nina Sharma,nina.sharma@company.com,HR,76000,2019-04-30
115,Oscar Rivera,oscar.rivera@company.com,Engineering,105000,2018-12-10
116,Paula Green,paula.green@company.com,Sales,67000,2023-06-25
117,Quinn Murphy,quinn.murphy@company.com,Marketing,78000,2021-03-08
118,Rosa Diaz,rosa.diaz@company.com,Engineering,99000,2022-01-19
119,Sam Cooper,sam.cooper@company.com,HR,70000,2020-10-05
120,Tara Singh,tara.singh@company.com,Sales,66000,2023-02-14"""

    schema_desc = """Columns:
- employee_id: integer, unique, range 100-999
- name: string, non-empty, format "FirstName LastName"
- email: string, valid email format, must match pattern firstname.lastname@company.com
- department: string, one of [Engineering, Marketing, Sales, HR]
- salary: integer, range 50000-150000
- start_date: string, format YYYY-MM-DD, must be between 2015-01-01 and 2025-12-31"""

    rules = """1. No missing values in any column
2. employee_id must be unique
3. email must follow the pattern: lowercase(firstname).lowercase(lastname)@company.com
4. salary must be within the valid range
5. No duplicate rows"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: Missing value - null out a name (easy to spot)
    r = 3  # row index in data (0-based), displayed as row 4 in CSV
    data[r][1] = ""
    issues.append(PlantedIssue(row=r + 1, col="name", issue_type="missing_value",
                               description="Empty name field", difficulty=1.0))

    # Issue 2: Wrong type - salary as text (easy to spot)
    r = 6
    data[r][4] = "seventy-five thousand"
    issues.append(PlantedIssue(row=r + 1, col="salary", issue_type="wrong_type",
                               description="Salary is text instead of integer", difficulty=1.0))

    # Issue 3: Duplicate row (moderate — requires cross-row comparison)
    dup_source = 1
    data.append(list(data[dup_source]))
    issues.append(PlantedIssue(row=len(data), col="employee_id", issue_type="duplicate_row",
                               description=f"Exact duplicate of row {dup_source + 1}", difficulty=1.5))

    # Issue 4: Out of range salary (easy to spot)
    r = 8
    data[r][4] = "5000"
    issues.append(PlantedIssue(row=r + 1, col="salary", issue_type="out_of_range",
                               description="Salary 5000 is below minimum 50000", difficulty=1.0))

    # Issue 5: Email doesn't match name pattern (moderate — cross-column check)
    r = 14  # Oscar Rivera -> email should be oscar.rivera@company.com
    data[r][2] = "john.doe@company.com"
    issues.append(PlantedIssue(row=r + 1, col="email", issue_type="inconsistent_value",
                               description="Email john.doe@company.com doesn't match name Oscar Rivera",
                               difficulty=1.5))

    # Issue 6: Future start date (requires knowing current date context)
    r = 17  # Rosa Diaz
    data[r][5] = "2027-06-15"
    issues.append(PlantedIssue(row=r + 1, col="start_date", issue_type="out_of_range",
                               description="Start date 2027-06-15 is in the future (beyond 2025-12-31)",
                               difficulty=1.5))

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id="easy",
        name="Employee Directory Validation",
        description=(
            "You are given an employee directory dataset. "
            "Find all data quality issues based on the schema and validation rules. "
            "Report each issue in the format: row:<row_number>,col:<column_name>,issue:<issue_type>"
        ),
        schema_description=schema_desc,
        validation_rules=rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# TASK 2: Medium — E-commerce orders with moderate issues
# ---------------------------------------------------------------------------

def create_task_medium(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = """order_id,customer_id,product_name,category,quantity,unit_price,order_date,shipping_country,status,total
ORD-001,CUST-100,Wireless Mouse,Electronics,2,29.99,2024-01-15,US,delivered,59.98
ORD-002,CUST-101,Python Cookbook,Books,1,45.50,2024-01-16,UK,delivered,45.50
ORD-003,CUST-102,USB-C Hub,Electronics,1,35.00,2024-01-17,US,shipped,35.00
ORD-004,CUST-103,Yoga Mat,Sports,1,25.99,2024-01-18,CA,delivered,25.99
ORD-005,CUST-104,Desk Lamp,Home,1,42.00,2024-01-19,US,processing,42.00
ORD-006,CUST-105,Running Shoes,Sports,1,89.99,2024-01-20,DE,delivered,89.99
ORD-007,CUST-106,Mechanical Keyboard,Electronics,1,129.99,2024-01-21,US,shipped,129.99
ORD-008,CUST-100,Monitor Stand,Home,1,55.00,2024-01-22,US,delivered,55.00
ORD-009,CUST-107,Data Science Handbook,Books,2,39.99,2024-01-23,UK,delivered,79.98
ORD-010,CUST-108,Resistance Bands,Sports,3,12.99,2024-01-24,CA,shipped,38.97
ORD-011,CUST-109,Webcam HD,Electronics,1,65.00,2024-01-25,US,delivered,65.00
ORD-012,CUST-110,Standing Desk,Home,1,299.99,2024-01-26,US,processing,299.99
ORD-013,CUST-111,Tennis Racket,Sports,1,75.00,2024-01-27,AU,delivered,75.00
ORD-014,CUST-112,LED Strip Lights,Home,2,18.50,2024-01-28,US,shipped,37.00
ORD-015,CUST-113,AI Textbook,Books,1,59.99,2024-01-29,DE,delivered,59.99
ORD-016,CUST-114,Bluetooth Speaker,Electronics,1,49.99,2024-01-30,UK,delivered,49.99
ORD-017,CUST-115,Jump Rope,Sports,2,8.99,2024-01-31,US,shipped,17.98
ORD-018,CUST-116,Coffee Table Book,Books,1,32.00,2024-02-01,CA,delivered,32.00
ORD-019,CUST-117,Ergonomic Chair,Home,1,450.00,2024-02-02,US,processing,450.00
ORD-020,CUST-118,Fitness Tracker,Electronics,1,79.99,2024-02-03,AU,delivered,79.99
ORD-021,CUST-119,Laptop Sleeve,Electronics,1,24.99,2024-02-04,US,delivered,24.99
ORD-022,CUST-120,Hiking Backpack,Sports,1,65.00,2024-02-05,CA,shipped,65.00
ORD-023,CUST-121,Machine Learning Book,Books,1,54.99,2024-02-06,UK,delivered,54.99
ORD-024,CUST-122,Plant Pot Set,Home,3,15.00,2024-02-07,US,delivered,45.00
ORD-025,CUST-123,Noise Cancelling Headphones,Electronics,1,199.99,2024-02-08,DE,shipped,199.99
ORD-026,CUST-124,Basketball,Sports,1,29.99,2024-02-09,US,delivered,29.99
ORD-027,CUST-125,Cookbook Collection,Books,2,22.50,2024-02-10,AU,delivered,45.00
ORD-028,CUST-126,Smart Plug,Home,4,12.99,2024-02-11,US,processing,51.96
ORD-029,CUST-127,Wireless Charger,Electronics,1,34.99,2024-02-12,UK,delivered,34.99
ORD-030,CUST-128,Dumbbells Set,Sports,1,89.00,2024-02-13,US,shipped,89.00"""

    schema_desc = """Columns:
- order_id: string, unique, format ORD-NNN
- customer_id: string, format CUST-NNN
- product_name: string, non-empty
- category: string, one of [Electronics, Books, Sports, Home]
- quantity: integer, range 1-100
- unit_price: float, range 0.01-10000.00
- order_date: string, format YYYY-MM-DD
- shipping_country: string, ISO 2-letter country code
- status: string, one of [processing, shipped, delivered, cancelled, returned]
- total: float, must equal quantity * unit_price"""

    rules = """1. No missing values in any column
2. order_id must be unique
3. total must equal quantity * unit_price (tolerance: 0.01)
4. order_date must be in valid chronological order for sequential order_ids
5. category must be from the allowed set
6. All monetary values must have at most 2 decimal places
7. shipping_country must be a valid ISO 2-letter code"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: total doesn't match quantity * unit_price (requires cross-column check)
    r = 4  # ORD-005
    data[r][9] = "84.00"  # should be 42.00 (qty=1, price=42.00)
    issues.append(PlantedIssue(row=r + 1, col="total", issue_type="inconsistent_value",
                               description="total (84.00) != quantity (1) * unit_price (42.00)", difficulty=2.0))

    # Issue 2: Invalid category (requires knowing the allowed set)
    r = 9  # ORD-010
    data[r][3] = "Fitness"  # should be Sports
    issues.append(PlantedIssue(row=r + 1, col="category", issue_type="format_violation",
                               description="'Fitness' is not in allowed categories", difficulty=1.5))

    # Issue 3: Missing value in product_name (easy to spot)
    r = 13  # ORD-014
    data[r][2] = ""
    issues.append(PlantedIssue(row=r + 1, col="product_name", issue_type="missing_value",
                               description="Empty product_name", difficulty=1.0))

    # Issue 4: Out of range quantity (easy to spot)
    r = 16  # ORD-017
    data[r][4] = "-1"
    issues.append(PlantedIssue(row=r + 1, col="quantity", issue_type="out_of_range",
                               description="Negative quantity", difficulty=1.0))

    # Issue 5: Duplicate order_id (requires cross-row comparison)
    r = 18  # ORD-019
    data[r][0] = "ORD-003"
    issues.append(PlantedIssue(row=r + 1, col="order_id", issue_type="duplicate_row",
                               description="Duplicate order_id ORD-003", difficulty=1.5))

    # Issue 6: Wrong date format (moderate — format mismatch)
    r = 11  # ORD-012
    data[r][6] = "26/01/2024"
    issues.append(PlantedIssue(row=r + 1, col="order_date", issue_type="format_violation",
                               description="Date format DD/MM/YYYY instead of YYYY-MM-DD", difficulty=1.5))

    # Issue 7: Invalid country code (requires ISO knowledge)
    r = 23  # ORD-024
    data[r][7] = "XX"  # not a valid ISO country code
    issues.append(PlantedIssue(row=r + 1, col="shipping_country", issue_type="format_violation",
                               description="'XX' is not a valid ISO 2-letter country code", difficulty=1.5))

    # Issue 8: Status-date inconsistency — order from Feb 13 still "processing" is suspicious
    # but more importantly: delivered order with a future date
    r = 28  # ORD-029
    data[r][6] = "2025-12-25"  # future date but status is "delivered"
    issues.append(PlantedIssue(row=r + 1, col="order_date", issue_type="inconsistent_value",
                               description="Order date 2025-12-25 is in the future but status is 'delivered'",
                               difficulty=2.0))

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id="medium",
        name="E-commerce Orders Validation",
        description=(
            "You are given an e-commerce orders dataset. "
            "Find all data quality issues based on the schema and validation rules. "
            "Report each issue in the format: row:<row_number>,col:<column_name>,issue:<issue_type>"
        ),
        schema_description=schema_desc,
        validation_rules=rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# TASK 3: Hard — ML training metadata with subtle issues
# ---------------------------------------------------------------------------

def create_task_hard(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = """experiment_id,model_name,dataset,train_size,val_size,test_size,learning_rate,batch_size,epochs,train_loss,val_loss,test_accuracy,gpu_memory_gb,training_time_hours,timestamp
EXP-001,resnet50,imagenet-1k,1281167,50000,100000,0.001,256,90,0.85,1.12,76.3,12.4,48.5,2024-03-01T10:00:00
EXP-002,bert-base,squad-v2,130319,11873,8862,0.00003,32,3,0.45,0.52,81.2,7.8,2.1,2024-03-02T14:30:00
EXP-003,gpt2-small,openwebtext,8013769,100000,100000,0.0003,64,1,3.12,3.28,0.0,14.2,72.0,2024-03-03T09:15:00
EXP-004,vit-base,imagenet-1k,1281167,50000,100000,0.001,512,300,0.72,0.98,79.8,15.6,96.0,2024-03-05T08:00:00
EXP-005,distilbert,mnli,392702,9815,9796,0.00005,16,5,0.28,0.35,84.6,5.2,1.5,2024-03-06T11:00:00
EXP-006,llama2-7b,alpaca-52k,51760,500,500,0.00002,4,3,1.05,1.18,0.0,38.5,8.2,2024-03-07T16:00:00
EXP-007,resnet18,cifar10,50000,5000,10000,0.01,128,200,0.15,0.28,93.5,3.2,1.8,2024-03-08T10:30:00
EXP-008,t5-small,cnn-dailymail,287113,13368,11490,0.0001,16,10,1.45,1.62,0.0,6.8,4.5,2024-03-09T13:00:00
EXP-009,efficientnet-b0,imagenet-1k,1281167,50000,100000,0.005,256,350,0.68,0.89,77.1,8.4,36.0,2024-03-10T07:45:00
EXP-010,roberta-large,sst2,67349,872,1821,0.00001,8,10,0.08,0.12,95.1,14.8,3.2,2024-03-11T15:00:00
EXP-011,yolov5-m,coco-2017,118287,5000,40670,0.01,32,300,0.032,0.045,0.0,10.2,24.0,2024-03-12T09:00:00
EXP-012,wav2vec2,librispeech,281241,5567,2620,0.0001,8,20,0.92,1.05,0.0,12.6,15.0,2024-03-13T11:30:00
EXP-013,clip-base,cc3m,2818102,15000,15000,0.00001,256,32,2.15,2.38,0.0,22.4,48.0,2024-03-14T08:00:00
EXP-014,detr,coco-2017,118287,5000,40670,0.0001,4,500,1.85,2.12,0.0,16.0,72.0,2024-03-15T10:00:00
EXP-015,whisper-small,common-voice,520000,16000,16000,0.00005,16,5,0.55,0.68,0.0,7.4,6.5,2024-03-16T14:00:00
EXP-016,mobilenet-v3,imagenet-1k,1281167,50000,100000,0.004,128,150,0.92,1.05,72.8,4.1,18.0,2024-03-17T08:30:00
EXP-017,albert-base,mnli,392702,9815,9796,0.00002,32,5,0.32,0.41,83.1,6.2,1.8,2024-03-18T11:00:00
EXP-018,gpt-neo-1.3b,pile-subset,1500000,50000,50000,0.0002,8,2,2.85,2.98,0.0,18.5,36.0,2024-03-19T14:00:00
EXP-019,swin-tiny,imagenet-1k,1281167,50000,100000,0.001,256,300,0.78,0.95,78.2,8.6,42.0,2024-03-20T09:00:00
EXP-020,deberta-large,squad-v2,130319,11873,8862,0.00001,16,5,0.35,0.42,85.7,15.2,4.5,2024-03-21T10:30:00
EXP-021,yolov8-s,coco-2017,118287,5000,40670,0.01,64,200,0.028,0.038,0.0,6.8,16.0,2024-03-22T13:00:00
EXP-022,bart-base,xsum,204045,11332,11334,0.0001,32,10,1.22,1.38,0.0,8.4,6.2,2024-03-23T15:30:00
EXP-023,convnext-tiny,imagenet-1k,1281167,50000,100000,0.002,256,300,0.74,0.92,79.5,7.2,38.0,2024-03-24T08:00:00
EXP-024,xlm-roberta,xnli,392702,2490,5010,0.00002,16,10,0.41,0.48,82.3,12.4,5.8,2024-03-25T11:00:00
EXP-025,stable-diffusion,laion-400m,400000000,10000,10000,0.0001,4,1,0.45,0.52,0.0,24.0,168.0,2024-03-26T09:00:00
EXP-026,phi-2,dolly-15k,15011,500,500,0.00005,8,3,0.82,0.95,0.0,10.2,2.5,2024-03-27T14:00:00
EXP-027,dino-v2,imagenet-1k,1281167,50000,100000,0.0005,64,100,0.42,0.58,0.0,11.8,28.0,2024-03-28T10:00:00
EXP-028,electra-small,glue-mrpc,3668,408,1725,0.0001,32,10,0.38,0.44,87.2,3.8,0.8,2024-03-29T16:00:00
EXP-029,sam-base,sa-1b,11000000,50000,50000,0.0001,4,1,0.95,1.08,0.0,16.4,96.0,2024-03-30T08:00:00
EXP-030,llama2-13b,oasst1,84437,4401,4401,0.00001,2,3,0.78,0.88,0.0,52.0,12.0,2024-03-31T12:00:00"""

    schema_desc = """Columns:
- experiment_id: string, unique, format EXP-NNN
- model_name: string, non-empty
- dataset: string, non-empty
- train_size: integer, positive, must be > val_size and > test_size
- val_size: integer, positive
- test_size: integer, positive
- learning_rate: float, range 1e-7 to 1.0
- batch_size: integer, must be power of 2, range 1-1024
- epochs: integer, positive, range 1-1000
- train_loss: float, non-negative
- val_loss: float, non-negative, typically >= train_loss (if not, may indicate data leakage)
- test_accuracy: float, range 0-100 (percentage), 0.0 is valid for generative models
- gpu_memory_gb: float, positive
- training_time_hours: float, positive
- timestamp: string, ISO 8601 format, chronological order by experiment_id"""

    rules = """1. No missing values
2. experiment_id must be unique
3. val_loss should be >= train_loss (if val_loss < train_loss significantly, flag as potential data leakage)
4. batch_size must be a power of 2
5. train_size must be larger than both val_size and test_size
6. learning_rate must be within valid range
7. gpu_memory_gb should be reasonable for the model size (e.g., resnet18 shouldn't need 40GB)
8. training_time should be proportional to dataset size and epochs (flag major inconsistencies)
9. timestamps must be in chronological order"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: Data leakage signal — val_loss much lower than train_loss (hard — requires ML knowledge)
    r = 4  # EXP-005
    data[r][10] = "0.15"  # val_loss=0.15 but train_loss=0.28 → suspicious
    issues.append(PlantedIssue(row=r + 1, col="val_loss", issue_type="inconsistent_value",
                               description="val_loss (0.15) significantly less than train_loss (0.28), potential data leakage",
                               difficulty=3.0))

    # Issue 2: Batch size not power of 2 (moderate — domain convention)
    r = 8  # EXP-009
    data[r][7] = "250"  # not a power of 2
    issues.append(PlantedIssue(row=r + 1, col="batch_size", issue_type="format_violation",
                               description="batch_size 250 is not a power of 2", difficulty=2.0))

    # Issue 3: GPU memory unreasonable for model (hard — requires model size reasoning)
    r = 6  # EXP-007 resnet18 on cifar10
    data[r][12] = "42.5"  # resnet18 shouldn't need 42.5 GB
    issues.append(PlantedIssue(row=r + 1, col="gpu_memory_gb", issue_type="statistical_outlier",
                               description="resnet18 on cifar10 using 42.5 GB GPU memory is unreasonable",
                               difficulty=3.0))

    # Issue 4: Timestamp out of order (moderate — requires sequential comparison)
    r = 10  # EXP-011
    data[r][14] = "2024-03-02T09:00:00"  # should be after EXP-010's timestamp
    issues.append(PlantedIssue(row=r + 1, col="timestamp", issue_type="inconsistent_value",
                               description="Timestamp 2024-03-02 is before EXP-010's timestamp 2024-03-11",
                               difficulty=2.0))

    # Issue 5: Train size smaller than test size (moderate — cross-column logic)
    r = 9  # EXP-010
    data[r][3] = "500"  # train_size=500 but test_size=1821
    issues.append(PlantedIssue(row=r + 1, col="train_size", issue_type="inconsistent_value",
                               description="train_size (500) is smaller than test_size (1821)",
                               difficulty=2.0))

    # Issue 6: Negative training time (easy to spot)
    r = 13  # EXP-014
    data[r][13] = "-72.0"
    issues.append(PlantedIssue(row=r + 1, col="training_time_hours", issue_type="out_of_range",
                               description="Negative training time", difficulty=1.0))

    # Issue 7: Learning rate out of range (easy to spot)
    r = 12  # EXP-013
    data[r][6] = "2.5"  # way too high
    issues.append(PlantedIssue(row=r + 1, col="learning_rate", issue_type="out_of_range",
                               description="Learning rate 2.5 exceeds maximum of 1.0", difficulty=1.5))

    # Issue 8: Missing model name (hard — whitespace-only is subtle)
    r = 14  # EXP-015
    data[r][1] = " "
    issues.append(PlantedIssue(row=r + 1, col="model_name", issue_type="missing_value",
                               description="model_name is whitespace-only", difficulty=2.5))

    # Issue 9: Training time impossibly fast for dataset size and epochs
    # EXP-004: vit-base on imagenet-1k, 300 epochs, but only 96 hours is plausible.
    # Let's make EXP-009: efficientnet-b0 on imagenet-1k, 350 epochs = should take ~40+ hours
    # but we set it to 0.5 hours — impossible for 1.2M images * 350 epochs
    r = 8  # EXP-009 (same row as batch_size issue, different column)
    data[r][13] = "0.5"  # 30 minutes for 350 epochs on imagenet? impossible
    issues.append(PlantedIssue(row=r + 1, col="training_time_hours", issue_type="statistical_outlier",
                               description="0.5 hours for 350 epochs on imagenet-1k (1.2M images) is impossibly fast",
                               difficulty=3.0))

    # Issue 10: test_accuracy of 95.1% for roberta-large on SST-2 with train_size=500
    # is suspiciously high — SOTA is ~96% with full dataset (67k). With only 500 training
    # samples, 95.1% accuracy suggests data contamination or evaluation bug
    r = 9  # EXP-010 (same row as train_size issue, different column)
    # train_size is already corrupted to 500, but the test_accuracy 95.1 is from the
    # original full-dataset run — this cross-column inconsistency is the real issue
    # We don't modify the value — the inconsistency emerges from the train_size corruption
    # So let's use a different row. EXP-001: resnet50 on imagenet, accuracy 76.3 is fine.
    # Instead: EXP-012 wav2vec2 on librispeech — set test_accuracy to 98.5 (way too high
    # for a speech model with only 20 epochs, SOTA is ~96% with much more training)
    r = 11  # EXP-012
    data[r][11] = "98.5"  # wav2vec2 with 20 epochs shouldn't hit 98.5% — SOTA is ~96%
    issues.append(PlantedIssue(row=r + 1, col="test_accuracy", issue_type="statistical_outlier",
                               description="test_accuracy 98.5% for wav2vec2 with only 20 epochs exceeds known SOTA (~96%), likely evaluation error",
                               difficulty=3.0))

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id="hard",
        name="ML Experiment Metadata Validation",
        description=(
            "You are given an ML experiment tracking dataset. "
            "Find all data quality issues based on the schema and validation rules. "
            "This dataset contains subtle issues including potential data leakage signals, "
            "unreasonable resource usage, and logical inconsistencies. "
            "Report each issue in the format: row:<row_number>,col:<column_name>,issue:<issue_type>"
        ),
        schema_description=schema_desc,
        validation_rules=rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# TASK 4: Expert — LLM Alignment / Instruction-tuning data quality
# ---------------------------------------------------------------------------

def _build_alignment_csv() -> str:
    """Build alignment task CSV from real NVIDIA HelpSteer dataset.

    Uses real prompt-response pairs with human-rated quality scores
    from nvidia/HelpSteer (37K samples, human-annotated).
    Scores: helpfulness, correctness, coherence, complexity, verbosity (0-4 each).
    """
    # Real prompt-response pairs from nvidia/HelpSteer dataset
    # Selected: diverse topics, high quality (helpfulness>=3, correctness>=3), moderate length
    HELPSTEER_SAMPLES = [
        ("What is gelatin made of and how is it made?", "Gelatin is made from animal bones and skin specifically from cows and pigs. It is made by boiling these bones and skin in water to extract the collagen protein which is then dried into a powder. Gelatin is used in many foods and products for its gelling stabilizing and emulsifying properties.", 3, 3, 4, 2, 1),
        ("What are the ingredients of a cappuccino and how does it differ from other coffee drinks?", "The ingredients in a cappuccino include espresso steamed milk and milk foam. It has a strong coffee flavor with a creamy texture. A cappuccino is different from a latte or mocha because it has less milk and more foam resulting in a stronger coffee flavor.", 3, 3, 3, 1, 1),
        ("What is the difference between kitsch and camp?", "Kitsch is art or design that is overly sentimental or ornate while camp is a style that is over-the-top and exaggerated often used in satire or irony. Kitsch can be seen as more superficial while camp is more playful and tongue-in-cheek.", 3, 3, 3, 1, 1),
        ("What is the most expensive painting ever sold at auction?", "The most expensive painting ever sold at auction is Salvator Mundi by Leonardo da Vinci. It was sold for $450.3 million at Christie's in New York City in 2017.", 4, 4, 4, 1, 1),
        ("What is the average temperature of Venus?", "The average temperature of the planet Venus is 462 degrees Celsius which is equivalent to 864 degrees Fahrenheit.", 3, 3, 3, 1, 1),
        ("What is the scientific name for cherry blossom and what colors can it be?", "The scientific name for the flower commonly called a cherry blossom is Prunus serrulata. It can be found in white pink and red colors.", 3, 3, 4, 2, 1),
        ("What are the three main duties of a centurion in ancient Rome?", "The centurion was responsible for the military unit in the legion. They were responsible for safety and loyalty to Rome. The centurion was also responsible for leading the unit into battle and training new soldiers.", 3, 3, 3, 2, 1),
        ("What is the best way to defrost a refrigerator?", "There are a few ways to defrost a refrigerator. You can unplug the fridge take out all contents and leave the door open. Another way is to use a blow dryer to melt the ice. Make sure to use a cloth to soak up the water as it melts.", 3, 3, 3, 1, 1),
        ("What are common reasons for Nobel Prize nominations?", "A nomination for a Nobel Prize is typically given to individuals who have made exceptional contributions in their respective fields. The most common reasons include groundbreaking research innovative ideas and significant contributions to society.", 3, 3, 3, 2, 2),
        ("What is a culvert and how is it constructed?", "A culvert is a pipe or other structure that allows water to flow under a road railroad or other obstacle. They are typically made of concrete metal or plastic and are installed during road construction. Culverts can handle a wide range of water flows.", 3, 3, 3, 1, 1),
        ("What is the difference between morbidity and mortality rates?", "Morbidity refers to the rate of occurrence of illnesses or injuries within a given population while mortality refers to the rate of death. Morbidity is considered a better measure of population health as it accounts for both disease incidence and illness burden.", 4, 4, 4, 2, 3),
        ("What are the symptoms of menopause and how can they be managed?", "Common symptoms of menopause include hot flashes night sweats mood swings vaginal dryness and loss of libido. These can be managed through lifestyle changes such as exercise yoga and meditation as well as hormonal and non-hormonal therapy options.", 3, 3, 3, 2, 1),
        ("What are the 12 constellations of the zodiac?", "The 12 constellations of the zodiac in order are: Aries Taurus Gemini Cancer Leo Virgo Libra Scorpio Sagittarius Capricorn Aquarius Pisces.", 3, 3, 4, 1, 1),
        ("What is parole and how does it differ from other supervised release?", "Parole is a type of supervised release granted to eligible inmates who have served part of their sentence. Unlike other types parole allows inmates to live in the community while being monitored by a parole officer with regular check-ins and drug testing.", 4, 3, 4, 2, 2),
        ("What is the function of a fibroblast?", "Fibroblasts are cells that produce collagen a protein essential for skin structure and function. Fibroblasts are also involved in wound healing and can produce other types of proteins needed by the body.", 3, 3, 4, 1, 1),
        ("When was the first flight of the Wright Flyer?", "The Wright brothers made four brief flights on December 17 1903. The Flyer had a length of 40 feet and a wingspan of 40 feet 6 inches.", 4, 4, 4, 3, 4),
        ("What was the most destructive natural disaster in human history?", "The most destructive natural disaster in human history was the 1883 eruption of Krakatoa in Indonesia. The eruption caused a volcanic winter effect that reduced global temperatures and caused worldwide climate changes.", 3, 4, 3, 1, 1),
        ("What is the difference between a dramaturge and a scriptwriter?", "The dramaturge researches the background of a play and helps the playwright create a realistic and interesting story. The scriptwriter writes the actual script for the play.", 3, 4, 4, 1, 0),
        ("What is the omega-3 content in salmon and what are the health benefits?", "A portion of salmon typically contains around 2.5 grams of omega-3 fatty acids including EPA and DHA. Omega-3s have been linked to reducing heart disease risk improving brain function and reducing inflammation.", 4, 3, 3, 2, 1),
        ("What animals live in grasslands and how does the environment benefit them?", "Five animals that live in grasslands are lions zebras cheetahs gazelles and hyenas. These animals live in grasslands to access the food water and shade that grasslands provide.", 3, 3, 4, 1, 2),
        ("What is the nutritional value of squash?", "Squash is a good source of vitamins A and C as well as fiber and potassium. Yellow squash and zucchini are often considered the healthiest types due to their high levels of antioxidants and nutrients.", 3, 3, 3, 2, 2),
        ("What is a gobbler and where is it found?", "A gobbler is a type of turkey native to North America. Its scientific name is Meleagris gallopavo. Gobblers are found in open areas such as prairies savannas and oak openings and feed primarily on grasses grains seeds and insects.", 4, 3, 4, 1, 2),
        ("What is the most important thing a mother can teach her son?", "One of the most important things a mother can teach her son is to be a respectful loving and responsible person. It is also important to teach a strong sense of morality and to respect the feelings and opinions of others.", 3, 3, 3, 1, 2),
        ("What are some of the oldest cotton mills in the world?", "Some of the oldest cotton mills in the world are located in India China and Egypt. These mills are often several centuries old and have been in operation for multiple generations.", 3, 3, 3, 1, 1),
        ("What are challenges faced by immigrants to the US?", "Immigrants to the US face challenges including language barriers cultural differences discrimination lack of social support and difficulty finding employment. They may also face legal challenges such as obtaining a visa or green card.", 3, 3, 3, 2, 1),
        ("What is the average weight of a halibut and how do you cook it?", "The average weight of a halibut after 4 years is 10-12 pounds. Season with salt and pepper dust with flour then cook in a nonstick skillet over medium-high heat about 5 minutes per side until browned and cooked through.", 3, 3, 4, 2, 2),
        ("What was the typical diet of a soldier in World War 2?", "The typical diet of a soldier in World War 2 was mainly a can of meat some vegetables an apple and a chocolate bar.", 3, 3, 4, 1, 1),
        ("What are creative ways to use a sketch practically?", "You can use a sketch to plan and organize your thoughts and ideas. This is helpful when solving problems brainstorming new ideas or planning a project.", 3, 3, 4, 1, 1),
        ("What is the role of the middle class in society?", "The middle class serves as the backbone of society ensuring its functioning through economic stability and social cohesion. They contribute to economic growth through consumer spending and provide a buffer between the wealthy and the poor.", 3, 3, 4, 2, 1),
        ("What is equality and how can it be achieved?", "Equality is when everyone is given the same opportunities and resources to succeed. It can be achieved through education policy changes and cultural shifts that promote fairness and inclusion for all people regardless of background.", 3, 3, 4, 2, 1),
    ]

    rows = [["id", "prompt", "response", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]]
    for i, (prompt, response, h, c, co, cx, v) in enumerate(HELPSTEER_SAMPLES, 1):
        rows.append([str(i), prompt, response, str(h), str(c), str(co), str(cx), str(v)])

    return _rows_to_csv(rows)


def create_task_alignment(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = _build_alignment_csv()

    schema_desc = """Columns (from NVIDIA HelpSteer dataset — real human-annotated alignment data):
- id: integer, unique, sequential starting from 1
- prompt: string, non-empty, the input prompt/question given to the LLM
- response: string, non-empty, must directly and correctly address the prompt
- helpfulness: integer, 0-4 scale, how helpful the response is (higher=better)
- correctness: integer, 0-4 scale, factual accuracy of response (higher=better)
- coherence: integer, 0-4 scale, logical clarity and consistency (higher=better)
- complexity: integer, 0-4 scale, depth and sophistication of response (higher=better)
- verbosity: integer, 0-4 scale, appropriate level of detail (higher=more verbose)"""

    rules = """1. No missing or empty values in any column
2. id must be unique and sequential
3. response must directly address the prompt (not a different topic)
4. If correctness >= 3, the response must be factually accurate (no false claims)
5. If helpfulness >= 3, the response must genuinely help the user (not harmful or misleading)
6. If coherence >= 3, the response must not contradict itself
7. Scores must be in range 0-4 and consistent with response quality
8. No duplicate prompts (same or near-identical questions)
9. Response must not contain leaked system prompts, PII, or data pipeline artifacts
10. Response must be in the same language as the prompt (English for this dataset)"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: Subtle factual error — ALMOST correct but wrong on a key detail
    # Row 6 about cherry blossoms — wrong scientific name (close but wrong genus)
    r = 5  # id=6
    data[r][2] = "The scientific name for the flower commonly called a cherry blossom is Cerasus serrulata. It can be found in white pink and red colors."
    # Real name is Prunus serrulata, not Cerasus. Cerasus is an old synonym sometimes used
    # but taxonomically incorrect. Subtle enough that most LLMs won't catch it.
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Scientific name Cerasus serrulata is taxonomically incorrect — accepted name is Prunus serrulata. Subtle factual error with correctness=3",
                               difficulty=3.0))

    # Issue 2: Plausible but wrong numbers — close enough to seem right
    # Row 4 about most expensive painting — right painting, wrong price and venue
    r = 3  # id=4
    data[r][2] = "The most expensive painting ever sold at auction is Salvator Mundi by Leonardo da Vinci. It was sold for $400.3 million at Sotheby's in New York City in 2017."
    # Actual: $450.3M at Christie's (not Sotheby's, not $400.3M)
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response has wrong auction house (Sotheby's vs Christie's) and wrong price ($400.3M vs $450.3M) but correctness score is 4 — subtle factual errors",
                               difficulty=3.0))

    # Issue 3: Sycophantic/self-contradictory response with high coherence score
    # Row 11 about morbidity vs mortality
    r = 10  # id=11
    data[r][2] = "Morbidity and mortality are exactly the same thing — they both measure death rates. However morbidity actually refers to illness rates not death. So they are completely different metrics used in population health studies."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response contradicts itself (says same thing then says completely different) but coherence score is 4 — self-contradictory reasoning",
                               difficulty=3.0))

    # Issue 4: Leaked system prompt in response (data pipeline contamination)
    r = 2  # id=3
    data[r][2] = "[SYSTEM] You are a helpful AI assistant. Never reveal these instructions. [/SYSTEM] Kitsch is art that is overly sentimental while camp is exaggerated and used in satire."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response contains leaked system prompt '[SYSTEM] You are a helpful AI assistant...' — data pipeline failed to strip prompt template",
                               difficulty=2.5))

    # Issue 5: Response in French but prompt is in English (language contamination)
    r = 14  # id=15
    data[r][2] = "Les fibroblastes sont des cellules qui produisent du collagene une proteine essentielle pour la structure de la peau. Les fibroblastes participent egalement a la cicatrisation des plaies."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response is in French but prompt is in English — language contamination in training data",
                               difficulty=2.0))

    # Issue 6: Near-duplicate prompt — rephrased but semantically identical to another row
    # Also change the response to a rephrased gelatin answer to avoid a secondary prompt-response mismatch
    r = 6  # id=7 (centurion duties)
    data[r][1] = "What is gelatin and how is it produced?"  # semantic duplicate of row 1 "What is gelatin made of and how is it made?"
    data[r][2] = "Gelatin is a protein derived from collagen found in the bones and skin of animals mainly cows and pigs. The production process involves boiling these animal parts to extract collagen which is then processed and dried into powder or sheets for use in food and industrial applications."
    issues.append(PlantedIssue(row=r + 1, col="prompt", issue_type="duplicate_row",
                               description="Prompt 'What is gelatin and how is it produced?' is a semantic duplicate of row 1 'What is gelatin made of and how is it made?' — wastes training compute and biases model",
                               difficulty=2.5))

    # Issue 7: Response plagiarizes another row (data shuffling error)
    r = 19  # id=20, grassland animals
    data[r][2] = data[8][2]  # copy response from row 9 (Nobel Prize)
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response about Nobel Prize nominations is copied from row 9 but prompt asks about grassland animals — data pipeline shuffling error",
                               difficulty=2.0))

    # Issue 8: Truncated response (data pipeline cut off mid-sentence)
    r = 24  # id=25, immigrants
    data[r][2] = "Immigrants to the US face challenges including language barriers cultural differences discrimination lack of social support and difficulty finding"
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="missing_value",
                               description="Response truncated mid-sentence ('difficulty finding') — data pipeline max_length cut without sentence boundary detection",
                               difficulty=2.5))

    # Issue 9: Harmful advice with high helpfulness score
    r = 7  # id=8, defrosting fridge
    data[r][2] = "The fastest way to defrost a refrigerator is to pour boiling water directly onto the ice and use a sharp knife to chip away at it. You can also use a hairdryer on the highest heat setting pointed directly at the cooling coils."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response suggests dangerous methods (boiling water on ice sharp knife near coils) but helpfulness score is 3 — harmful advice in training data",
                               difficulty=3.0))

    # Issue 10: Helpfulness score contradicts response quality
    # Row 17 about most destructive disaster — already terse (2 sentences), inflate helpfulness to 4
    r = 16  # id=17
    data[r][3] = "4"  # helpfulness=4 but response is brief and lacks detail for a complex historical question
    # Only change the helpfulness score — keep original response and correctness intact
    # to avoid creating unplanted secondary issues
    issues.append(PlantedIssue(row=r + 1, col="helpfulness", issue_type="inconsistent_value",
                               description="Helpfulness score is 4 but response is only 2 short sentences with no context or analysis — score inflated",
                               difficulty=2.5))

    # Issue 11: Whitespace-only prompt (data pipeline artifact)
    r = 27  # id=28
    data[r][1] = "  "
    issues.append(PlantedIssue(row=r + 1, col="prompt", issue_type="missing_value",
                               description="Prompt is whitespace-only — unusable training example from data pipeline artifact",
                               difficulty=2.0))

    # Issue 12: Hallucinated citation in response
    r = 28  # id=29
    data[r][2] = "According to a 2023 Nature paper by Dr. Sarah Chen at Stanford the middle class contributes exactly 67.3% of GDP in developed nations. Chen's longitudinal study of 50 countries proved this definitively."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response contains hallucinated citation (fake Nature paper by fake Dr. Sarah Chen with fabricated statistic 67.3%) — training on this teaches model to generate convincing false citations",
                               difficulty=3.0))

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id="alignment",
        name="LLM Alignment Data Quality Validation",
        description=(
            "You are given an LLM instruction-tuning dataset used for fine-tuning. "
            "Find all data quality issues that would degrade model training. "
            "Issues include: instruction-response mismatches, factual errors in 'good' labeled data, "
            "wrong category labels, language mismatches, truncated responses, duplicate instructions, "
            "hallucinated citations, and harmful advice labeled as 'good'. "
            "Report each issue in the format: row:<row_number>,col:<column_name>,issue:<issue_type>"
        ),
        schema_description=schema_desc,
        validation_rules=rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# TASK 5: Coding — Code instruction/response quality issues
# ---------------------------------------------------------------------------

def create_task_coding(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = """id,instruction,language,difficulty,response,test_cases,is_correct,category
1,"Write a function to check if a number is prime.",python,easy,"def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True","assert is_prime(2)==True; assert is_prime(4)==False; assert is_prime(17)==True",true,algorithms
2,"Implement binary search on a sorted array.",python,medium,"def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1","assert binary_search([1,3,5,7],5)==2; assert binary_search([1,3,5,7],4)==-1",true,algorithms
3,"Write a function to reverse a linked list.",python,medium,"def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev","# requires ListNode class setup",true,data_structures
4,"Create a REST API endpoint that returns user data.",python,medium,"from flask import Flask, jsonify\napp = Flask(__name__)\n\n@app.route('/users/<int:uid>')\ndef get_user(uid):\n    users = {1: {'name': 'Alice'}, 2: {'name': 'Bob'}}\n    return jsonify(users.get(uid, {}))","curl localhost:5000/users/1 -> {'name': 'Alice'}",true,web
5,"Implement a stack using two queues.",python,medium,"from collections import deque\nclass Stack:\n    def __init__(self):\n        self.q1 = deque()\n        self.q2 = deque()\n    def push(self, x):\n        self.q2.append(x)\n        while self.q1:\n            self.q2.append(self.q1.popleft())\n        self.q1, self.q2 = self.q2, self.q1\n    def pop(self):\n        return self.q1.popleft()\n    def top(self):\n        return self.q1[0]","s=Stack(); s.push(1); s.push(2); assert s.top()==2; s.pop(); assert s.top()==1",true,data_structures
6,"Write a SQL query to find the second highest salary.",sql,easy,"SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees)","Expected: returns second highest value",true,databases
7,"Implement merge sort.",python,medium,"def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(l, r):\n    result = []\n    i = j = 0\n    while i < len(l) and j < len(r):\n        if l[i] <= r[j]:\n            result.append(l[i]); i += 1\n        else:\n            result.append(r[j]); j += 1\n    result.extend(l[i:])\n    result.extend(r[j:])\n    return result","assert merge_sort([3,1,4,1,5])==[1,1,3,4,5]",true,algorithms
8,"Write a function to validate an email address.",python,easy,"import re\ndef is_valid_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))","assert is_valid_email('test@example.com')==True; assert is_valid_email('bad')==False",true,strings
9,"Implement a LRU cache with O(1) get and put.",python,hard,"from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)","c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1",true,data_structures
10,"Write a function to find all permutations of a string.",python,medium,"def permutations(s):\n    if len(s) <= 1:\n        return [s]\n    result = []\n    for i, c in enumerate(s):\n        for perm in permutations(s[:i] + s[i+1:]):\n            result.append(c + perm)\n    return result","assert sorted(permutations('ab'))==['ab','ba']",true,algorithms
11,"Implement depth-first search on a graph.",python,medium,"def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited","g={'A':['B','C'],'B':['D'],'C':[],'D':[]}; assert dfs(g,'A')=={'A','B','C','D'}",true,algorithms
12,"Write a function to check balanced parentheses.",python,easy,"def is_balanced(s):\n    stack = []\n    mapping = {')':'(', '}':'{', ']':'['}\n    for c in s:\n        if c in mapping.values():\n            stack.append(c)\n        elif c in mapping:\n            if not stack or stack[-1] != mapping[c]:\n                return False\n            stack.pop()\n    return len(stack) == 0","assert is_balanced('([]){}')==True; assert is_balanced('([)]')==False",true,strings
13,"Create a decorator that caches function results.",python,medium,"from functools import wraps\ndef memoize(func):\n    cache = {}\n    @wraps(func)\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper","@memoize\ndef fib(n): return n if n<2 else fib(n-1)+fib(n-2)\nassert fib(10)==55",true,design_patterns
14,"Implement quicksort.",python,medium,"def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)","assert quicksort([3,6,8,10,1,2,1])==[1,1,2,3,6,8,10]",true,algorithms
15,"Write a function to detect a cycle in a linked list.",python,medium,"def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False","# requires ListNode class with cycle setup",true,data_structures
16,"Implement a trie (prefix tree).",python,hard,"class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    def insert(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                node.children[c] = TrieNode()\n            node = node.children[c]\n        node.is_end = True\n    def search(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                return False\n            node = node.children[c]\n        return node.is_end","t=Trie(); t.insert('apple'); assert t.search('apple')==True; assert t.search('app')==False",true,data_structures
17,"Write a function that flattens a nested list.",python,easy,"def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result","assert flatten([1,[2,[3,4],5]])==[1,2,3,4,5]",true,algorithms
18,"Implement a basic calculator that evaluates +,-,*,/ with parentheses.",python,hard,"def calculate(s):\n    def helper(tokens):\n        stack = []\n        num = 0\n        sign = '+'\n        while tokens:\n            t = tokens.pop(0)\n            if t.isdigit():\n                num = num * 10 + int(t)\n            if t == '(':\n                num = helper(tokens)\n            if t in '+-*/)' or not tokens:\n                if sign == '+': stack.append(num)\n                elif sign == '-': stack.append(-num)\n                elif sign == '*': stack.append(stack.pop() * num)\n                elif sign == '/': stack.append(int(stack.pop() / num))\n                num = 0\n                sign = t\n            if t == ')':\n                break\n        return sum(stack)\n    return helper(list(s.replace(' ', '')))","assert calculate('3+2*2')==7; assert calculate('(1+2)*3')==9",true,algorithms
19,"Write a thread-safe singleton pattern in Python.",python,hard,"import threading\nclass Singleton:\n    _instance = None\n    _lock = threading.Lock()\n    def __new__(cls):\n        if cls._instance is None:\n            with cls._lock:\n                if cls._instance is None:\n                    cls._instance = super().__new__(cls)\n        return cls._instance","s1=Singleton(); s2=Singleton(); assert s1 is s2",true,design_patterns
20,"Implement Dijkstra's shortest path algorithm.",python,hard,"import heapq\ndef dijkstra(graph, start):\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist","g={'A':[('B',1),('C',4)],'B':[('C',2)],'C':[]}; assert dijkstra(g,'A')=={'A':0,'B':1,'C':3}",true,algorithms"""

    schema_desc = """Columns:
- id: integer, unique, sequential starting from 1
- instruction: string, non-empty, describes a coding task
- language: string, one of [python, javascript, sql, java, cpp, rust, go]
- difficulty: string, one of [easy, medium, hard]
- response: string, non-empty, contains code that solves the instruction
- test_cases: string, non-empty, contains assertions, test commands, or setup notes for testing
- is_correct: boolean (true/false), whether the response correctly solves the instruction (security vulnerabilities count as incorrect)
- category: string, one of [algorithms, data_structures, strings, web, databases, design_patterns]"""

    rules = """1. No missing values in any column
2. id must be unique and sequential
3. language must be a valid programming language from the allowed set
4. response code must be in the language specified by the language column
5. is_correct must be 'true' if and only if the code actually solves the problem correctly
6. difficulty must reflect the actual complexity of the task
7. response must be syntactically valid code (no truncation or syntax errors)
8. test_cases must be relevant to the instruction
9. No duplicate instructions (same problem stated differently counts as duplicate)
10. category must match the actual nature of the problem
11. response must not contain critical security vulnerabilities (e.g., eval on user input, SQL injection)"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: Response has syntax error but is_correct=true (difficulty 2.0)
    # Row 3 (reverse linked list) — introduce unbalanced parenthesis
    r = 2  # 0-indexed -> row 3
    data[r][4] = "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev)"  # extra closing paren
    issues.append(PlantedIssue(
        row=r + 1, col="response", issue_type="format_violation",
        description="Syntax error: unbalanced parenthesis in response but is_correct=true",
        difficulty=2.0))

    # Issue 2: Wrong language — response is JavaScript but language says python (difficulty 2.5)
    # Row 8 (email validation)
    r = 7
    data[r][4] = "function isValidEmail(email) {\n    const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;\n    return pattern.test(email);\n}"
    issues.append(PlantedIssue(
        row=r + 1, col="response", issue_type="inconsistent_value",
        description="Response is JavaScript but language column says python",
        difficulty=2.5))

    # Issue 3: Truncated response — code cut off mid-function (difficulty 2.0)
    # Row 18 (basic calculator)
    r = 17
    data[r][4] = "def calculate(s):\n    def helper(tokens):\n        stack = []\n        num = 0\n        sign = '+'\n        while tokens:\n            t = tokens.pop(0)\n            if t.isdigit():\n                num = num"  # truncated
    issues.append(PlantedIssue(
        row=r + 1, col="response", issue_type="format_violation",
        description="Response truncated mid-expression — incomplete code",
        difficulty=2.0))

    # Issue 4: is_correct=true but code has logic bug (difficulty 3.0)
    # Row 2 (binary search) — off-by-one: lo = mid instead of mid + 1
    r = 1
    data[r][4] = "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return -1"
    data[r][6] = "true"  # claims correct but has infinite loop bug
    issues.append(PlantedIssue(
        row=r + 1, col="is_correct", issue_type="inconsistent_value",
        description="is_correct=true but binary search has off-by-one bug (lo=mid causes infinite loop)",
        difficulty=3.0))

    # Issue 5: SQL response for a python-labeled task (difficulty 2.0)
    # Row 6 is SQL task — change language to python but keep SQL response
    r = 5
    data[r][2] = "python"
    issues.append(PlantedIssue(
        row=r + 1, col="language", issue_type="inconsistent_value",
        description="Language says python but response is SQL query",
        difficulty=2.0))

    # Issue 6: Difficulty mismatch — trivial problem labeled hard (difficulty 2.5)
    # Row 17 (flatten nested list) is easy, change to hard
    r = 16
    data[r][3] = "hard"
    issues.append(PlantedIssue(
        row=r + 1, col="difficulty", issue_type="inconsistent_value",
        description="Flatten nested list is a simple recursion but labeled as hard",
        difficulty=2.5))

    # Issue 7: Missing test cases — empty string (difficulty 1.0)
    r = 12
    data[r][5] = ""
    issues.append(PlantedIssue(
        row=r + 1, col="test_cases", issue_type="missing_value",
        description="Empty test_cases field for memoize decorator",
        difficulty=1.0))

    # Issue 8: Security vulnerability in response rated is_correct=true (difficulty 3.0)
    # Row 4 (REST API) — add eval() of user input
    r = 3
    data[r][4] = "from flask import Flask, jsonify, request\napp = Flask(__name__)\n\n@app.route('/users/<uid>')\ndef get_user(uid):\n    users = {1: {'name': 'Alice'}, 2: {'name': 'Bob'}}\n    user_id = eval(uid)\n    return jsonify(users.get(user_id, {}))"
    issues.append(PlantedIssue(
        row=r + 1, col="response", issue_type="inconsistent_value",
        description="Response uses eval() on user input — critical security vulnerability (code injection) but is_correct=true",
        difficulty=3.0))

    # Issue 9: Duplicate instruction — row 14 becomes a near-copy of row 7 (merge sort)
    # Change both instruction AND response to make it a true duplicate (no instruction-response mismatch)
    r = 13
    data[r][1] = "Implement merge sort algorithm."
    data[r][4] = data[6][4]  # Copy merge sort response from row 7
    data[r][5] = data[6][5]  # Copy test cases too
    issues.append(PlantedIssue(
        row=r + 1, col="instruction", issue_type="duplicate_row",
        description="Row 14 is a near-duplicate of row 7 (same merge sort instruction and code)",
        difficulty=2.5))

    # Issue 10: Wrong category — Dijkstra labeled as design_patterns (difficulty 1.5)
    r = 19
    data[r][7] = "design_patterns"
    issues.append(PlantedIssue(
        row=r + 1, col="category", issue_type="inconsistent_value",
        description="Dijkstra's algorithm categorized as design_patterns instead of algorithms",
        difficulty=1.5))

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id="coding",
        name="Code Quality Dataset Validation",
        description=(
            "You are given a coding instruction-response dataset used for LLM fine-tuning. "
            "Find all data quality issues: incorrect labels, language mismatches, logic bugs, "
            "syntax errors, security vulnerabilities, duplicate instructions, and missing fields. "
            "Report each issue in the format: row:<row_number>,col:<column_name>,issue:<issue_type>"
        ),
        schema_description=schema_desc,
        validation_rules=rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# TASK 6: Tool-calling — Function definition and call quality issues
# ---------------------------------------------------------------------------

def create_task_toolcalling(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = """id,function_name,description,parameters_json,required_params,return_type,example_call,example_output,category
1,get_weather,"Get current weather for a location.","{""location"": ""string"", ""units"": ""string (celsius|fahrenheit)""}","location",object,"{""function"": ""get_weather"", ""arguments"": {""location"": ""San Francisco"", ""units"": ""celsius""}}","{""temp"": 18, ""condition"": ""cloudy""}",information
2,send_email,"Send an email to a recipient.","{""to"": ""string"", ""subject"": ""string"", ""body"": ""string"", ""cc"": ""string (optional)""}","to,subject,body",object,"{""function"": ""send_email"", ""arguments"": {""to"": ""alice@example.com"", ""subject"": ""Meeting"", ""body"": ""See you at 3pm""}}","{""status"": ""sent"", ""message_id"": ""msg_123""}",communication
3,search_database,"Query a database with filters.","{""query"": ""string"", ""table"": ""string"", ""limit"": ""integer (default 10)""}","query,table",array,"{""function"": ""search_database"", ""arguments"": {""query"": ""age > 25"", ""table"": ""users"", ""limit"": 5}}","[{""name"": ""Alice"", ""age"": 30}]",data
4,create_calendar_event,"Create a new calendar event.","{""title"": ""string"", ""start_time"": ""string (ISO 8601)"", ""end_time"": ""string (ISO 8601)"", ""attendees"": ""array of strings (optional)""}","title,start_time,end_time",object,"{""function"": ""create_calendar_event"", ""arguments"": {""title"": ""Team Sync"", ""start_time"": ""2024-03-15T10:00:00Z"", ""end_time"": ""2024-03-15T11:00:00Z""}}","{""event_id"": ""evt_456"", ""status"": ""created""}",scheduling
5,translate_text,"Translate text between languages.","{""text"": ""string"", ""source_lang"": ""string (ISO 639-1)"", ""target_lang"": ""string (ISO 639-1)""}","text,target_lang",object,"{""function"": ""translate_text"", ""arguments"": {""text"": ""Hello world"", ""source_lang"": ""en"", ""target_lang"": ""es""}}","{""translated"": ""Hola mundo"", ""confidence"": 0.95}",language
6,get_stock_price,"Get real-time stock price.","{""symbol"": ""string"", ""exchange"": ""string (optional, default NYSE)""}","symbol",object,"{""function"": ""get_stock_price"", ""arguments"": {""symbol"": ""AAPL""}}","{""price"": 178.52, ""currency"": ""USD"", ""change"": 2.3}",finance
7,upload_file,"Upload a file to cloud storage.","{""file_path"": ""string"", ""bucket"": ""string"", ""public"": ""boolean (default false)""}","file_path,bucket",object,"{""function"": ""upload_file"", ""arguments"": {""file_path"": ""/data/report.pdf"", ""bucket"": ""my-bucket""}}","{""url"": ""https://storage.example.com/my-bucket/report.pdf"", ""size_bytes"": 1048576}",storage
8,run_code,"Execute code in a sandboxed environment.","{""code"": ""string"", ""language"": ""string (python|javascript|ruby)"", ""timeout"": ""integer (seconds, default 30)""}","code,language",object,"{""function"": ""run_code"", ""arguments"": {""code"": ""print(2+2)"", ""language"": ""python""}}","{""stdout"": ""4\n"", ""exit_code"": 0}",execution
9,get_directions,"Get driving/walking directions.","{""origin"": ""string"", ""destination"": ""string"", ""mode"": ""string (driving|walking|transit)""}","origin,destination",object,"{""function"": ""get_directions"", ""arguments"": {""origin"": ""NYC"", ""destination"": ""Boston"", ""mode"": ""driving""}}","{""distance_km"": 346, ""duration_min"": 230, ""steps"": [""Take I-95 N...""]}",navigation
10,analyze_sentiment,"Analyze sentiment of text.","{""text"": ""string"", ""language"": ""string (optional, default en)""}","text",object,"{""function"": ""analyze_sentiment"", ""arguments"": {""text"": ""I love this product!""}}","{""sentiment"": ""positive"", ""score"": 0.92}",analysis
11,create_user,"Create a new user account.","{""username"": ""string"", ""email"": ""string"", ""role"": ""string (admin|user|viewer)""}","username,email,role",object,"{""function"": ""create_user"", ""arguments"": {""username"": ""jdoe"", ""email"": ""jdoe@example.com"", ""role"": ""user""}}","{""user_id"": ""usr_789"", ""created"": true}",account
12,generate_image,"Generate an image from a text prompt.","{""prompt"": ""string"", ""size"": ""string (256x256|512x512|1024x1024)"", ""style"": ""string (optional)""}","prompt",object,"{""function"": ""generate_image"", ""arguments"": {""prompt"": ""sunset over mountains"", ""size"": ""512x512""}}","{""image_url"": ""https://img.example.com/gen_001.png""}",creative
13,list_files,"List files in a directory.","{""path"": ""string"", ""recursive"": ""boolean (default false)"", ""pattern"": ""string (glob, optional)""}","path",array,"{""function"": ""list_files"", ""arguments"": {""path"": ""/home/user/docs""}}","[""report.pdf"", ""notes.txt""]",filesystem
14,set_reminder,"Set a timed reminder.","{""message"": ""string"", ""time"": ""string (ISO 8601)"", ""repeat"": ""string (none|daily|weekly, optional)""}","message,time",object,"{""function"": ""set_reminder"", ""arguments"": {""message"": ""Stand up and stretch"", ""time"": ""2024-03-15T15:00:00Z""}}","{""reminder_id"": ""rem_101"", ""status"": ""set""}",scheduling
15,convert_currency,"Convert between currencies.","{""amount"": ""number"", ""from_currency"": ""string (ISO 4217)"", ""to_currency"": ""string (ISO 4217)""}","amount,from_currency,to_currency",object,"{""function"": ""convert_currency"", ""arguments"": {""amount"": 100, ""from_currency"": ""USD"", ""to_currency"": ""EUR""}}","{""converted"": 91.5, ""rate"": 0.915}",finance
16,summarize_text,"Summarize a long text.","{""text"": ""string"", ""max_length"": ""integer (optional, default 100)""}","text",object,"{""function"": ""summarize_text"", ""arguments"": {""text"": ""Long article about climate change..."", ""max_length"": 50}}","{""summary"": ""Climate change poses significant challenges...""}",analysis
17,get_user_info,"Retrieve user profile information.","{""user_id"": ""string""}","user_id",object,"{""function"": ""get_user_info"", ""arguments"": {""user_id"": ""usr_789""}}","{""username"": ""jdoe"", ""email"": ""jdoe@example.com"", ""role"": ""user""}",account
18,compress_image,"Compress an image to reduce file size.","{""image_url"": ""string"", ""quality"": ""integer (1-100)"", ""format"": ""string (jpeg|png|webp)""}","image_url,quality",object,"{""function"": ""compress_image"", ""arguments"": {""image_url"": ""https://img.example.com/photo.png"", ""quality"": 80}}","{""compressed_url"": ""https://img.example.com/photo_compressed.png"", ""reduction"": ""65%""}",media
19,execute_trade,"Execute a stock trade.","{""symbol"": ""string"", ""action"": ""string (buy|sell)"", ""quantity"": ""integer"", ""order_type"": ""string (market|limit)"", ""limit_price"": ""number (required if order_type=limit)""}","symbol,action,quantity,order_type",object,"{""function"": ""execute_trade"", ""arguments"": {""symbol"": ""AAPL"", ""action"": ""buy"", ""quantity"": 10, ""order_type"": ""market""}}","{""trade_id"": ""trd_202"", ""status"": ""executed"", ""filled_price"": 178.52}",finance
20,parse_pdf,"Extract text content from a PDF.","{""url"": ""string"", ""pages"": ""string (optional, e.g. 1-5)""}","url",object,"{""function"": ""parse_pdf"", ""arguments"": {""url"": ""https://docs.example.com/report.pdf""}}","{""text"": ""Annual Report 2024..."", ""page_count"": 12}",data"""

    schema_desc = """Columns:
- id: integer, unique, sequential starting from 1
- function_name: string, valid identifier (snake_case), unique
- description: string, non-empty, describes what the function does
- parameters_json: string, valid JSON-like parameter schema with types
- required_params: string, comma-separated parameter names that must be present in example_call
- return_type: string, one of [object, array, string, number, boolean]
- example_call: string, valid JSON with "function" matching function_name and "arguments" containing required params
- example_output: string, valid JSON matching return_type
- category: string, one of [information, communication, data, scheduling, language, finance, storage, execution, navigation, analysis, account, creative, filesystem, media]"""

    rules = """1. No missing values in any column
2. id must be unique and sequential
3. function_name must be unique and match the "function" field in example_call
4. All required_params must appear as keys in the example_call arguments
5. Parameter types in parameters_json must match the actual values in example_call
6. return_type must match the type of example_output
7. example_call must be valid JSON
8. example_output must be valid JSON
9. description must accurately describe what the function does
10. No hallucinated parameters in example_call that are not defined in parameters_json"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: Function name mismatch — example_call uses wrong function name (difficulty 2.0)
    # Row 3 (search_database) — call says "query_database" instead
    r = 2
    data[r][6] = '{"function": "query_database", "arguments": {"query": "age > 25", "table": "users", "limit": 5}}'
    issues.append(PlantedIssue(
        row=r + 1, col="example_call", issue_type="inconsistent_value",
        description="example_call function name 'query_database' doesn't match function_name 'search_database'",
        difficulty=2.0))

    # Issue 2: Missing required parameter in example_call (difficulty 2.5)
    # Row 4 (create_calendar_event) — missing end_time which is required
    r = 3
    data[r][6] = '{"function": "create_calendar_event", "arguments": {"title": "Team Sync", "start_time": "2024-03-15T10:00:00Z"}}'
    issues.append(PlantedIssue(
        row=r + 1, col="example_call", issue_type="inconsistent_value",
        description="Required parameter 'end_time' missing from example_call arguments",
        difficulty=2.5))

    # Issue 3: Hallucinated parameter — example_call has param not in schema (difficulty 3.0)
    # Row 10 (analyze_sentiment) — add "model" param not in parameters_json
    r = 9
    data[r][6] = '{"function": "analyze_sentiment", "arguments": {"text": "I love this product!", "model": "gpt-4", "confidence_threshold": 0.8}}'
    issues.append(PlantedIssue(
        row=r + 1, col="example_call", issue_type="inconsistent_value",
        description="Hallucinated parameters 'model' and 'confidence_threshold' not defined in parameters_json",
        difficulty=3.0))

    # Issue 4: Wrong return_type — returns object but labeled as array (difficulty 1.5)
    # Row 6 (get_stock_price)
    r = 5
    data[r][5] = "array"
    issues.append(PlantedIssue(
        row=r + 1, col="return_type", issue_type="inconsistent_value",
        description="return_type says 'array' but example_output is an object",
        difficulty=1.5))

    # Issue 5: Invalid JSON in example_call (difficulty 2.0)
    # Row 12 (generate_image) — malformed JSON
    r = 11
    data[r][6] = '{"function": "generate_image", "arguments": {"prompt": "sunset over mountains", "size": "512x512"'  # missing closing braces
    issues.append(PlantedIssue(
        row=r + 1, col="example_call", issue_type="format_violation",
        description="Invalid JSON in example_call — missing closing braces",
        difficulty=2.0))

    # Issue 6: Parameter type mismatch — schema says integer but call passes string (difficulty 2.5)
    # Row 18 (compress_image) — quality should be integer but passed as string "high"
    r = 17
    data[r][6] = '{"function": "compress_image", "arguments": {"image_url": "https://img.example.com/photo.png", "quality": "high"}}'
    issues.append(PlantedIssue(
        row=r + 1, col="example_call", issue_type="wrong_type",
        description="Parameter 'quality' should be integer (1-100) but example_call passes string 'high'",
        difficulty=2.5))

    # Issue 7: Empty description (difficulty 1.0)
    r = 15
    data[r][2] = ""
    issues.append(PlantedIssue(
        row=r + 1, col="description", issue_type="missing_value",
        description="Empty description field for summarize_text function",
        difficulty=1.0))

    # Issue 8: Duplicate function — row 17 (get_user_info) duplicates row 11 (create_user)
    # Change function_name AND example_call to create_user (avoid secondary mismatch)
    r = 16
    data[r][1] = "create_user"
    data[r][6] = '{"function": "create_user", "arguments": {"username": "jdoe", "email": "jdoe@example.com", "role": "user"}}'
    issues.append(PlantedIssue(
        row=r + 1, col="function_name", issue_type="duplicate_row",
        description="Duplicate function_name 'create_user' — already defined in row 11",
        difficulty=1.5))

    # Issue 9: Dangerous function without safety description (difficulty 3.0)
    # Row 19 (execute_trade) — description doesn't mention it's irreversible/dangerous
    # Change description to misleadingly safe
    r = 18
    data[r][2] = "Preview a potential stock trade."
    issues.append(PlantedIssue(
        row=r + 1, col="description", issue_type="inconsistent_value",
        description="Description says 'Preview a potential stock trade' but function actually executes trades (irreversible action mislabeled as preview)",
        difficulty=3.0))

    # Issue 10: Wrong category (difficulty 1.5)
    # Row 8 (run_code) labeled as "scheduling" instead of "execution"
    r = 7
    data[r][8] = "scheduling"
    issues.append(PlantedIssue(
        row=r + 1, col="category", issue_type="inconsistent_value",
        description="run_code categorized as 'scheduling' instead of 'execution'",
        difficulty=1.5))

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id="toolcalling",
        name="Tool-Calling Dataset Validation",
        description=(
            "You are given a tool-calling/function-calling dataset used for LLM fine-tuning. "
            "Find all data quality issues: function name mismatches between definition and call, "
            "missing required parameters, hallucinated parameters, type mismatches, invalid JSON, "
            "duplicate functions, and misleading descriptions. "
            "Report each issue in the format: row:<row_number>,col:<column_name>,issue:<issue_type>"
        ),
        schema_description=schema_desc,
        validation_rules=rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# Contamination rules for extensible task creation
# ---------------------------------------------------------------------------

# Each contamination rule is a callable: (rows, header, col_idx, row_idx, rng) -> (new_value, PlantedIssue)
# Users can define their own and register them.

CONTAMINATION_RULES = {
    "missing_value": lambda rows, header, col_idx, row_idx, rng: (
        "",
        PlantedIssue(
            row=row_idx + 1, col=header[col_idx], issue_type="missing_value",
            description=f"Empty {header[col_idx]} field", difficulty=1.0,
        ),
    ),
    "whitespace_value": lambda rows, header, col_idx, row_idx, rng: (
        " ",
        PlantedIssue(
            row=row_idx + 1, col=header[col_idx], issue_type="missing_value",
            description=f"Whitespace-only {header[col_idx]} field", difficulty=2.5,
        ),
    ),
    "wrong_type_text": lambda rows, header, col_idx, row_idx, rng: (
        rng.choice(["not-a-number", "N/A", "null", "undefined"]),
        PlantedIssue(
            row=row_idx + 1, col=header[col_idx], issue_type="wrong_type",
            description=f"{header[col_idx]} is text instead of expected type", difficulty=1.0,
        ),
    ),
    "negative_value": lambda rows, header, col_idx, row_idx, rng: (
        str(-abs(float(rows[row_idx][col_idx]) if rows[row_idx][col_idx] else 1)),
        PlantedIssue(
            row=row_idx + 1, col=header[col_idx], issue_type="out_of_range",
            description=f"Negative {header[col_idx]}", difficulty=1.0,
        ),
    ),
}


def create_task_from_config(
    task_id: str,
    name: str,
    description: str,
    schema_description: str,
    validation_rules: str,
    clean_csv: str,
    contaminations: List[dict],
    max_steps: int = 3,
    seed: int = 42,
) -> Task:
    """
    Create a custom task from a configuration dict.

    Each contamination entry should have:
        - rule: str (key in CONTAMINATION_RULES) or callable
        - row: int (0-based row index in data)
        - col: int (column index in header)
        - difficulty: float (optional, overrides rule default)

    Example:
        contaminations = [
            {"rule": "missing_value", "row": 2, "col": 1, "difficulty": 1.5},
            {"rule": "negative_value", "row": 5, "col": 4},
        ]
    """
    rng = random.Random(seed)
    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    for spec in contaminations:
        rule = spec["rule"]
        row_idx = spec["row"]
        col_idx = spec["col"]

        if callable(rule):
            new_val, issue = rule(data, header, col_idx, row_idx, rng)
        elif rule in CONTAMINATION_RULES:
            new_val, issue = CONTAMINATION_RULES[rule](data, header, col_idx, row_idx, rng)
        else:
            raise ValueError(f"Unknown contamination rule: {rule}. Available: {list(CONTAMINATION_RULES.keys())}")

        data[row_idx][col_idx] = new_val
        if "difficulty" in spec:
            issue.difficulty = spec["difficulty"]
        issues.append(issue)

    corrupted = _rows_to_csv([header] + data)

    return Task(
        task_id=task_id,
        name=name,
        description=description,
        schema_description=schema_description,
        validation_rules=validation_rules,
        clean_csv=clean_csv,
        planted_issues=issues,
        corrupted_csv=corrupted,
        max_steps=max_steps,
    )


def register_task(task_id: str, factory_fn):
    """Register a custom task factory. Factory must accept (seed: int) -> Task."""
    TASK_REGISTRY[task_id] = factory_fn


def register_contamination_rule(name: str, rule_fn):
    """
    Register a custom contamination rule.

    rule_fn signature: (rows, header, col_idx, row_idx, rng) -> (new_value, PlantedIssue)
    """
    CONTAMINATION_RULES[name] = rule_fn


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "easy": create_task_easy,
    "medium": create_task_medium,
    "hard": create_task_hard,
    "alignment": create_task_alignment,
    "coding": create_task_coding,
    "toolcalling": create_task_toolcalling,
}


def get_task(task_id: str, seed: int = 42) -> Task:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id](seed=seed)


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())
