**Name:** Andrei-Bogdan Ciobanu

**Group:** 322CB
<br>

# Project PP 2022

*This documents represents a summary of the comments in the Tasks.hs source file.*
<br>

## Taskset 1 (Introduction)

### "Global" helper functions

    sum_row :: [String] -> Float

- *sum_row* calculates the sum of the elements of a row.
<br>

### Task 1 - 0.2p

    compute_average_steps :: Table -> Table

- *compute_average_steps* calculates the average steps of each person in a table by summing the steps for each hour and then dividing it by 8 (eight hours);
- *(tail m)* -> ignores the table header.
<br>

### Task 2 - 0.2p

    get_passed_people_num :: Table -> Int

- *check_steps* sums the steps of each person (*sum_row (tail p)*) and compares the value with the goal (1000 steps);
- *(tail m)* -> ignores the table header;
- *(tail p)* -> ignores the person's name.

<br>

    get_passed_people_percentage :: Table -> Float

- the percentage of people who have achieved their goal is obtained by dividing the number of people who have achieved their daily goal (*get_passed_people_num m*) by the total number of people (*length m - 1*).

<br>

    get_steps_avg :: Table -> Float

- *get_steps_avg* calculates the total number of steps made by all people with a foldr, and divides it by the number of people (*length m - 1*), in order to get the average number of daily steps;
- *(tail m)* -> ignores the table header.
<br>

### Task 3 - 0.2p

    get_steps_per_h :: Table -> Table

- *get_steps_per_h* takes the eight_hours table and returns a table with 8 rows and N columns, where N is the number of people; each row contains the steps each person took in that hour (H10 to H17).

<br>

    get_avg_steps_per_h :: Table -> Table

- *get_avg_steps_per_h* calculates the average steps for each hour by summing the steps on each row (*sum_row row*) and dividing the value by N, where N is the number of people (*length row*); a foldr is being used in order to build the list of values.
<br>

### Task 4 - 0.2p

    get_activ_minutes :: Table -> Table

- *get_activ_minutes* takes the *physical_activity* table and returns a table with the last 3 columns of *physical_activity* transposed table (a row of "very active minutes", a row of "fairly active minutes" and a row of "lightly active minutes").

<br>

    count_helper :: (Int -> Int -> Int) -> [String] -> String

- *count_helper* takes a function representing the "filter" of a range and a row of activity minutes, and returns the number of values that fit the given range;
- *(tail row)* -> ignores the kind of minutes: "VeryActiveMinutes", "FairlyActiveMinutes" or "LightlyActiveMinutes";
- *(read :: String -> Int)* -> converts a String to Int;
- *(show :: Int -> String)* -> converts an Int to String
- a *foldr* is used in order to process the row, apply the *range_func* function to each value and compute the accumulator, which represents how many minutes were spent being active based on the given range.

<br>

    get_activ_summary :: Table -> Table

- *get_activ_summary* makes use of the previous two helper functions; *map* is used to process each row of the table and build the resulting table;
- *count_range1* -> for the range [0, 50);
- *count_range2* -> for the range [50, 100);
- *count_range3* -> for the range [100, 500);
<br>

### Task 5 - 0.2p

    get_ranking :: Table -> Table

- *(tail m)* -> ignores the table header;
- *(take 2)* -> extracts the column with the name and the column with the total number of steps;
- *map* is used to apply *(take 2)* for each row of the table (excluding the table header), resulting a subtable with the name and total number of steps of each person;
- the subtable is then sorted by the total number of steps (ascending), followed by alphabetical name order (*cmp_func*).
<br>

### Task 6 - 0.2p

    calculate_average :: [String] -> Float

- *calculate_average* computes the average value of a row of values.

<br>

    get_avg_first_4h :: [String] -> Float

- *get_avg_first_4h* calculates the average steps for the first 4 hours of the day.

<br>

    get_avg_last_4h :: [String] -> Float

- *get_avg_last_4h* calculates the average steps for the last 4 hours of the day.

<br>

    get_difference :: [String] -> Float

- *get_difference* calculates the absolute difference between the number of steps taken in the two parts of the day.

<br>

    get_steps_diff_table :: Table -> Table

- *(tail m)* -> ignores the table header;
- applies *sortBy* on the table with 4 columns: “Name”, “Average first 4h”, “Average last 4h”, “Difference”;
- the table with the four columns mentioned above is built using *map* and *get_table* function;
- the table is sorted by the "Difference" column (ascending), followed by alphabetical name order (*cmp_func*).
<br>

### Task 7 - 0.1p

    vmap :: (Value -> Value) -> Table -> Table

- applies the given function to all the values, by using *map*.
<br>

### Task 8 - 0.1p

    rmap :: (Row -> Row) -> [String] -> Table -> Table

- applies the given function to all the entries, by using *map*;
- the *[String]* variable is the new table header.

<br>

    get_sleep_total :: Row -> Row

- *get_sleep_total* takes a row from *sleep_min* table and returns a row with two values: email and total number of minutes slept that week;
- the total number of minutes slept that week is calculated with *sum_row*.
<br><br>

## Taskset 2

### "Global" helper functions

    get_column_number :: ColumnName -> Table -> Int

- *get_column_number* returns the column number in the table (starting from 0);
- searching is done in the table header.
<br>

### Task 1 - 0.2p

    tsort :: ColumnName -> Table -> Table

- *tsort* takes a column name and a table and returns the table sorted by that column (if multiple entries have the same values, then it is sorted by the first column);
- *(tail table)* -> ignores the table header;
- *get_row_element* returns the element in the given column;
- *cmp_func* -> determines the type of the values;
- *cmp_str* -> ascending order for numeric values;
- *cmp_float* -> lexicographic order for strings.
<br>

### Task 2 - 0.1p

    vunion :: Table -> Table -> Table

- *vunion* takes two tables and adds all rows from the second table at the end of the first table, if column names coincide; if columns names are not the same, the first table remains unchanged.
<br>

### Task 3 - 0.2p

    hunion :: Table -> Table -> Table

- *hunion* takes two tables and extends each row of the first table with a row of the second table (simple horizontal union of 2 tables); if one of the tables has more lines than the other, the shorter table is padded with rows containing only empty strings;
- *(replicate (length (head t1)) "")* -> padding for the first table;
- *(replicate (length (head t2)) "")* -> padding for the second table;
- *zip_with_padding* -> does the extension for each row and applies padding if necessary.
<br>

### Task 4 - 0.3p

    tjoin :: ColumnName -> Table -> Table -> Table
<br>

### Task 5 - 0.2p

    cartesian :: (Row -> Row -> Row) -> [ColumnName] -> Table -> Table -> Table

- *cartesian* takes two tables and applies the given operation on each entry in the first table with each entry in the second table => the cartesian product of the two tables.
<br>

### Task 6 - 0.2p

    projection :: [ColumnName] -> Table -> Table

- *projection* extracts from the table those columns specified by the names in given list of column names;
- each row in the table is processed with a *foldr*;
- *extract* -> used by *foldr*; extracts by indexing (!!) the columns specified by the names in the list of column names, and appends the results to the accumulator.
<br>

### Task 7 - 0.2p

    filterTable :: (Value -> Bool) -> ColumnName -> Table -> Table

- *filterTable* filters rows based on a given condition applied to a specified column;
- each row in the table is processed with a *foldr*;
- *filter_row* -> if the column in the row meets the condition, then the row is appended to the accumulator; otherwise, the row is not appended to the accumulator.
<br>
<br>

## Taskset 3

### Task 1 - 0.3p

- enrolls `Query` in class `Eval` (without `Filter` or `Graph`);

<br>

    extract_column :: ColumnName -> Table -> [Value]

- returns column *colname* as a list.
<br>

### Task 2 - 0.2p

- enrolls `FilterCondition` in class `FEval` and implements `eval` for `Filter` query;

<br>

    filter_aux :: FEval a => FilterCondition a -> Table -> Table

- filters rows based on a given *FilterCondition*.
<br>

### Task 3 - 0.2p

- implements `eval` for `Graph` query;

<br>

    create_graph :: EdgeOp -> Table -> Table

- creates a graph from *Table t*;
- the weight of an edge between two nodes is given by *edgeop*.
<br>

### Task 4 - 0.3p

- extracts similarity graph;

<br>

    similarities_query :: Query

- “From” and “To” are users’ names and "Value" is the distance between the 2 users’ hours slept;
- the distance between two users is "the sum of intervals where both users slept an equal amount";
- keeps only the rows with distance >= 5;
- the edges in the resulting graph are sorted by the "Value" column.
<br>
