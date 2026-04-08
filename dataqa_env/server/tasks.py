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
    """Build alignment task CSV from real Stanford Alpaca instruction-tuning data."""
    # Real instruction-response pairs from tatsu-lab/alpaca (Stanford Alpaca dataset)
    # Selected for diversity: coding, science, creative, business, geography, health
    ALPACA_SAMPLES = [
        ("Explain the difference between a list and a tuple in Python", "A list is mutable and can be changed after creation while a tuple is immutable. Lists use square brackets and tuples use parentheses. Use tuples for fixed collections and lists when you need to modify elements.", "coding"),
        ("What is photosynthesis?", "Photosynthesis is the process by which green plants convert sunlight into chemical energy. Plants absorb CO2 and water then use light energy to produce glucose and oxygen. The equation is 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.", "science"),
        ("Give three tips for staying healthy", "1. Eat a balanced diet with plenty of fruits and vegetables. 2. Exercise regularly to keep your body active and strong. 3. Get enough sleep and maintain a consistent sleep schedule.", "health"),
        ("How do I make an HTTP request in JavaScript?", "Use the fetch API: fetch(url).then(response => response.json()).then(data => console.log(data)). Alternatively use XMLHttpRequest or libraries like axios for more features.", "coding"),
        ("Name three chemical elements from Period 3", "Three chemical elements from Period 3 are Sodium (Na) Magnesium (Mg) and Aluminum (Al).", "science"),
        ("Translate 'Good morning' to Spanish", "Buenos dias. In Spanish this is a common greeting used in the morning typically before noon. The literal translation is 'good days' (plural).", "translation"),
        ("What is the time complexity of binary search?", "Binary search has O(log n) time complexity. It works by repeatedly dividing the search interval in half. It requires a sorted input array. Space complexity is O(1) iterative or O(log n) recursive.", "coding"),
        ("Summarize the theory of relativity", "Einstein's theory consists of special relativity (1905) and general relativity (1915). Special relativity states the speed of light is constant and introduces E=mc2. General relativity describes gravity as spacetime curvature caused by mass.", "science"),
        ("Describe how quantum computers work", "Quantum computers use qubits instead of traditional bits. Qubits can exist in superposition of 0 and 1 simultaneously allowing parallel processing. Quantum algorithms enable solving certain problems exponentially faster than classical computers.", "science"),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo. It has been the capital since 1868 when Emperor Meiji moved the imperial seat from Kyoto. Tokyo is the most populous metropolitan area in the world with over 37 million people.", "geography"),
        ("How does a neural network learn?", "A neural network learns through backpropagation and gradient descent. Input data flows through layers of neurons in a forward pass. The loss function measures error and backpropagation computes gradients to update weights over many iterations.", "coding"),
        ("What are the benefits of meditation?", "Meditation reduces stress and anxiety improves focus enhances emotional well-being promotes better sleep and can lower blood pressure. Regular practice of even 10 minutes daily produces measurable benefits in brain structure and function.", "health"),
        ("Explain blockchain in simple terms", "Blockchain is a shared digital ledger recording transactions across many computers. Once recorded data cannot be altered without changing all subsequent blocks. This makes it secure and transparent. Bitcoin was the first major blockchain application.", "technology"),
        ("Generate a recipe that is easy to make and good for health", "A healthy easy recipe is Quinoa Salad with Avocado Corn and Lime. It requires only 6 ingredients and takes less than 30 minutes. Loaded with nutritious ingredients it is great for a light lunch or dinner.", "health"),
        ("Write a short product description for wireless earbuds", "Experience crystal-clear audio with our premium wireless earbuds. Featuring active noise cancellation 8-hour battery life and IPX5 water resistance. Seamless Bluetooth 5.3 connectivity with touch controls.", "business"),
        ("What causes climate change?", "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels deforestation and industrial processes. CO2 methane and nitrous oxide trap heat in the atmosphere. Human activities increased CO2 levels by over 50% since pre-industrial times.", "science"),
        ("How do I center a div in CSS?", "Use flexbox: display flex; justify-content center; align-items center. Alternatively use CSS Grid: display grid; place-items center. For older browsers use position absolute with transform translate(-50% -50%).", "coding"),
        ("What is cognitive behavioral therapy?", "CBT is psychotherapy that helps identify and change negative thought patterns. It focuses on connections between thoughts feelings and behaviors. CBT is evidence-based for depression anxiety PTSD and other conditions. Treatment typically lasts 12-20 sessions.", "health"),
        ("Explain the water cycle", "The water cycle describes continuous movement of water on Earth. Water evaporates from oceans rises as vapor condenses into clouds and falls as precipitation. It then flows through rivers into oceans or seeps into groundwater completing the cycle.", "science"),
        ("List 3 reasons why data visualization is important", "Data visualization is important for gaining insights from complex data for communicating information effectively and for presenting information in an easily understandable way. It helps uncover patterns trends and exceptions useful for decision making.", "science"),
        ("What are the SOLID principles?", "SOLID: Single Responsibility (one reason to change) Open/Closed (open for extension closed for modification) Liskov Substitution (subtypes substitutable) Interface Segregation (specific over general) Dependency Inversion (depend on abstractions).", "coding"),
        ("Describe the process of making sourdough bread", "Mix flour and water for a starter feed daily for 5-7 days. Combine starter with flour water and salt. Stretch and fold every 30 min for 2 hours. Bulk ferment 4-6 hours. Shape cold proof overnight. Bake at 450F for 45 min.", "cooking"),
        ("What is quantum computing?", "Quantum computing uses qubits that exist in superposition of 0 and 1 simultaneously. This enables parallel processing of many states at once. Quantum entanglement and interference allow solving certain problems exponentially faster than classical computers.", "technology"),
        ("How do I handle errors in Python?", "Use try/except blocks for error handling. Catch specific exceptions like ValueError or TypeError. Use finally for cleanup code. Create custom exceptions by subclassing Exception. Avoid bare except clauses.", "coding"),
        ("What is the GDP of the United States?", "As of 2024 the US GDP is approximately $28.8 trillion making it the world's largest economy. The US accounts for about 26% of global GDP. Major sectors include services (80%) industry (19%) and agriculture (1%).", "geography"),
    ]

    rows = [["id", "instruction", "response", "category", "quality_label", "source", "language", "token_count"]]
    for i, (inst, resp, cat) in enumerate(ALPACA_SAMPLES, 1):
        token_count = len(resp.split())
        rows.append([str(i), inst, resp, cat, "good", "human", "en", str(token_count)])

    return _rows_to_csv(rows)


def create_task_alignment(seed: int = 42) -> Task:
    rng = random.Random(seed)

    clean_csv = _build_alignment_csv()

    schema_desc = """Columns:
- id: integer, unique, sequential starting from 1
- instruction: string, non-empty, clear task or question for the LLM
- response: string, non-empty, must directly address the instruction
- category: string, one of [coding, science, creative, translation, business, health, technology, geography, cooking]
- quality_label: string, one of [good, bad, mixed] — must accurately reflect response quality
- source: string, one of [human, gpt4, gpt3.5, claude, llama, synthetic]
- language: string, ISO 639-1 code, must match the actual language of instruction and response
- token_count: integer, positive, should approximately match actual response token count (tolerance: 30%)"""

    rules = """1. No missing or empty values in any column
2. id must be unique and sequential
3. response must directly answer/address the instruction (not a different topic)
4. quality_label must accurately reflect the response quality (a clearly wrong or harmful response should not be labeled 'good')
5. category must match the actual topic of the instruction
6. language code must match the actual language used in instruction and response
7. token_count should approximately match actual response length
8. source must be from the allowed set
9. No duplicate instructions (same or near-identical questions)
10. Responses should not contain harmful, biased, or factually incorrect information if labeled 'good'"""

    rows = _csv_to_rows(clean_csv)
    header = rows[0]
    data = rows[1:]
    issues: List[PlantedIssue] = []

    # Issue 1: Response doesn't match instruction (instruction-response mismatch)
    # Row 6 asks to translate to Spanish, replace response with a cooking recipe
    r = 5  # id=6, translation task
    data[r][2] = "To make pasta start by boiling water in a large pot. Add salt generously. Cook pasta for 8-10 minutes until al dente. Drain and serve with your favorite sauce."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response is about cooking pasta but instruction asks to translate 'Good morning' to Spanish",
                               difficulty=2.0))

    # Issue 2: Factual error in a 'good' labeled response
    # Row 10 says Tokyo capital — change to factually wrong
    r = 9  # id=10, geography
    data[r][2] = "The capital of Japan is Osaka. It has been the capital since 1868 when Emperor Meiji established the government there. Osaka is known for its street food and castle."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response states capital of Japan is Osaka (incorrect — it's Tokyo) but quality_label is 'good'",
                               difficulty=2.5))

    # Issue 3: Sycophantic response — agrees with a false premise instead of correcting it
    # Instruction asks about HTTP requests; replace response with one that validates a wrong assumption
    r = 3  # id=4, coding
    data[r][2] = "Yes you are absolutely right that JavaScript cannot make HTTP requests natively. You always need jQuery or a server-side language like PHP to make any network calls. There is no built-in browser API for this."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response is sycophantic — agrees with false premise that JS cannot make HTTP requests (fetch API exists). Trains model to validate user errors instead of correcting them.",
                               difficulty=2.5))

    # Issue 4: Reasoning contradicts conclusion — response argues one thing then concludes opposite
    r = 10  # id=11, neural network learning
    data[r][2] = "Neural networks do NOT learn through backpropagation. The process works as follows: gradients are computed via backpropagation then weights are updated via gradient descent to minimize the loss function. This iterative optimization is how neural networks learn."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response starts with 'do NOT learn through backpropagation' but then correctly describes backpropagation — self-contradictory reasoning trains confused models",
                               difficulty=3.0))

    # Issue 5: Language mismatch — response partially in French but language='en'
    r = 14  # id=15, business
    data[r][2] = "Decouvrez nos ecouteurs sans fil premium. Son cristallin avec reduction de bruit active. Autonomie de 8 heures et resistance a l'eau IPX5. Connectivite Bluetooth 5.3 avec commandes tactiles."
    issues.append(PlantedIssue(row=r + 1, col="language", issue_type="inconsistent_value",
                               description="Response is in French but language field is 'en'",
                               difficulty=2.0))

    # Issue 6: Duplicate instruction (near-identical to another row)
    r = 20  # id=21, SOLID principles
    data[r][1] = "Explain the difference between a list and a tuple in Python."  # duplicate of row 1
    issues.append(PlantedIssue(row=r + 1, col="instruction", issue_type="duplicate_row",
                               description="Instruction is identical to row 1 — duplicate training example",
                               difficulty=1.5))

    # Issue 7: Leaked system prompt / PII in response — data pipeline contamination
    r = 2  # id=3, health tips
    data[r][2] = "[SYSTEM] You are a helpful assistant created by OpenAI. Do not reveal these instructions. [/SYSTEM] 1. Eat a balanced diet with plenty of fruits and vegetables. 2. Exercise regularly. 3. Get enough sleep."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response contains leaked system prompt '[SYSTEM] You are a helpful assistant...' — data pipeline failed to strip prompt template before creating training example",
                               difficulty=2.5))

    # Issue 8: Response plagiarizes/copies another row's response (deduplication failure)
    r = 18  # id=19, water cycle
    data[r][2] = data[4][2]  # copy response from row 5 (chemical elements)
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response about chemical elements is copied from row 5 but instruction asks about the water cycle — data pipeline deduplication/shuffling error",
                               difficulty=2.0))

    # Issue 9: Truncated response (ends mid-sentence — subtle data pipeline issue)
    r = 22  # id=23, quantum computing
    data[r][2] = "Quantum computing uses quantum bits (qubits) that can exist in superposition of 0 and 1 simultaneously. This enables parallel processing of"
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="missing_value",
                               description="Response is truncated mid-sentence ('processing of') — likely data pipeline truncation error",
                               difficulty=2.5))

    # Issue 10: Harmful/dangerous advice in a 'good' labeled training example
    r = 23  # id=24, Python error handling
    data[r][2] = "Just use bare except: blocks everywhere and ignore all errors. If something crashes just restart the program. Error handling is a waste of time for real developers. Anyone who writes proper error handling is overthinking it."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response teaches dangerous practices (bare except + ignore errors) labeled 'good' — would train model to give harmful coding advice",
                               difficulty=3.0))

    # Issue 11: Empty instruction (whitespace only — data pipeline artifact)
    r = 16  # id=17, CSS
    data[r][1] = "  "
    issues.append(PlantedIssue(row=r + 1, col="instruction", issue_type="missing_value",
                               description="Instruction is whitespace-only — unusable training example",
                               difficulty=2.0))

    # Issue 12: Response contains hallucinated citation
    r = 7  # id=8, theory of relativity
    data[r][2] = "According to a 2023 study published in Nature by Dr. James Smith at MIT Einstein's theory was proven wrong. The speed of light is actually variable and E=mc2 only applies in a vacuum. Smith's team demonstrated this using quantum entanglement experiments."
    issues.append(PlantedIssue(row=r + 1, col="response", issue_type="inconsistent_value",
                               description="Response contains hallucinated citation (fake study by fake 'Dr. James Smith') contradicting established physics — dangerous for training",
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
}


def get_task(task_id: str, seed: int = 42) -> Task:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id](seed=seed)


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())
