-- =============== DO NOT MODIFY ===================
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Use camelCase" #-}

-- ==================================================

module Tasks where

import Common
import Data.Array
import Data.List
import Data.Maybe
import Dataset
import Text.Printf
import Text.Read

type CSV = String

type Value = String

type Row = [Value]

type Table = [Row]

type ColumnName = String

-- Prerequisities
split_by :: Char -> String -> [String]
split_by x = foldr op [""]
  where
    op char acc
      | char == x = "" : acc
      | otherwise = (char : head (acc)) : tail (acc)

read_csv :: CSV -> Table
read_csv = (map (split_by ',')) . (split_by '\n')

write_csv :: Table -> CSV
write_csv =
  (foldr (++) [])
    . (intersperse "\n")
    . (map (foldr (++) []))
    . (map (intersperse ","))

{-
    TASK SET 1
-}

-- sum_row calculates the sum of the elements of a row
sum_row :: [String] -> Float
sum_row row = sum (map (read :: String -> Float) row)

-- Task 1

-- compute_average_steps calculates the average steps of each person in a table by summing the steps for each hour and
-- then dividing it by 8 (eight hours)
compute_average_steps :: Table -> Table
compute_average_steps m =
  ["Name", "Average Number of Steps"] :
  map average (tail m) -- (tail m) -> to ignore the table header
  where
    average p = [head p, printf "%.2f" (sum_row (tail p) / 8)]

-- Task 2

-- Number of people who have achieved their goal:
-- check_steps sums the steps of each person and compares the value with the goal (1000 steps)
get_passed_people_num :: Table -> Int
get_passed_people_num m = foldr check_steps 0 (tail m) -- (tail m) -> to ignore the table header
  where
    check_steps p acc
      | sum_row (tail p) >= 1000 = acc + 1 -- (tail p) -> to ignore the person's name
      | otherwise = acc

-- Percentage of people who have achieved their goal:
get_passed_people_percentage :: Table -> Float
get_passed_people_percentage m = fromIntegral (get_passed_people_num m) / fromIntegral (length m - 1)

-- Average number of daily steps
-- get_steps_avg calculates the total number of steps made by all people with a foldr, and divides it by the number of
-- people, in order to get the average number of daily steps
get_steps_avg :: Table -> Float
get_steps_avg m = foldr get_steps 0 (tail m) / fromIntegral (length m - 1) -- (tail m) -> to ignore the table header
  where
    get_steps p acc = acc + sum_row (tail p)

-- Task 3

-- get_steps_per_h takes the eight_hours table and returns a table with 8 rows and N columns, where N is the number of
-- people; each row contains the steps each person took in that hour (H10 to H17)
get_steps_per_h :: Table -> Table
get_steps_per_h m = tail (transpose (tail m)) -- (tail m) -> to ignore the table header

-- get_avg_steps_per_h calculates the average steps for each hour by summing the steps on each row and dividing the
-- value by N, where N is the number of people; a foldr is being used in order to build the list of values
get_avg_steps_per_h :: Table -> Table
get_avg_steps_per_h m = ["H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17"] : [foldr avg [] (get_steps_per_h m)]
  where
    avg row acc = printf "%.2f" (sum_row row / fromIntegral (length row)) : acc

-- Task 4

-- get_activ_minutes takes the physical_activity table and returns a table with the last 3 columns of physical_activity
-- transposed table (a row of "very active minutes", a row of "fairly active minutes" and a row of "lightly active
-- minutes")
get_activ_minutes :: Table -> Table
get_activ_minutes m = drop 3 (transpose m)

-- count_helper takes a function representing the "filter" of a range and a row of activity minutes, and returns the
-- number of values that fit the given range;
-- (tail row) -> ignores the kind of minutes: "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes";
-- (read :: String -> Int) -> converts a String to an Int;
-- a foldr is used in order to process the row, apply the range_func function to each value and compute the accumulator,
-- which represents how many minutes were spent being active based on the given range.
count_helper :: (Int -> Int -> Int) -> [String] -> String
count_helper range_func row = (show :: Int -> String) (foldr (range_func . (read :: String -> Int)) 0 (tail row))

-- get_activ_summary makes use of the previous two helper functions; map is used to process each row of the table and
-- build the resulting table
get_activ_summary :: Table -> Table
get_activ_summary m = ["column", "range1", "range2", "range3"] : map count (get_activ_minutes m)
  where
    count row = [head row, count_helper count_range1 row, count_helper count_range2 row, count_helper count_range3 row]
      where
        count_range1 mins acc
          | mins >= 0 && mins < 50 = acc + 1
          | otherwise = acc

        count_range2 mins acc
          | mins >= 50 && mins < 100 = acc + 1
          | otherwise = acc

        count_range3 mins acc
          | mins >= 100 && mins < 500 = acc + 1
          | otherwise = acc

-- Task 5

-- (tail m) -> to ignore the table header
-- (take 2) -> extracts the column with the name and the column with the total number of steps
-- map is used to apply (take 2) for each row of the table (excluding the table header)
get_ranking :: Table -> Table
get_ranking m = ["Name", "Total Steps"] : sortBy cmp_func (map (take 2) (tail m))
  where
    cmp_func row1 row2 -- by their total number of steps (ascending), followed by alphabetical name order
      | (read (last row1) :: Int) < (read (last row2) :: Int) = LT
      | (read (last row1) :: Int) > (read (last row2) :: Int) = GT
      | otherwise = compare (head row1) (head row2)

-- Task 6

-- calculate_average computes the average value of a row of values
calculate_average :: [String] -> Float
calculate_average row = sum_row row / fromIntegral (length row)

-- get_avg_first_4h calculates the average steps for the first 4 hours of the day
get_avg_first_4h :: [String] -> Float
get_avg_first_4h p = calculate_average (take 4 (tail p)) -- (tail p) -> to ignore the name of the person

-- get_avg_last_4h calculates the average steps for the last 4 hours of the day
get_avg_last_4h :: [String] -> Float
get_avg_last_4h p = calculate_average (drop 4 (tail p)) -- (tail p) -> to ignore the name of the person

-- get_difference calculates the absolute difference between the number of steps taken in the two parts of the day
get_difference :: [String] -> Float
get_difference p = abs (get_avg_first_4h p - get_avg_last_4h p)

-- (tail m) -> to ignore the table header
-- applies sortBy on the table with 4 columns: “Name”, “Average first 4h”, “Average last 4h”, “Difference”
-- the table with the four columns mentioned above is built using map and get_table function
get_steps_diff_table :: Table -> Table
get_steps_diff_table m =
  ["Name", "Average first 4h", "Average last 4h", "Difference"] :
  sortBy cmp_func (map get_table (tail m)) -- (tail m) -> to ignore the table header
  where
    get_table p =
      [ head p,
        printf "%.2f" (get_avg_first_4h p),
        printf "%.2f" (get_avg_last_4h p),
        printf "%.2f" (get_difference p)
      ]

    cmp_func row1 row2 -- by the "Difference" column (ascending), followed by alphabetical name order
      | (read (last row1) :: Float) < (read (last row2) :: Float) = LT
      | (read (last row1) :: Float) > (read (last row2) :: Float) = GT
      | otherwise = compare (head row1) (head row2)

-- Task 7

-- Applies the given function to all the values
vmap :: (Value -> Value) -> Table -> Table
vmap f = map (map f)

-- Task 8

-- Applies the given function to all the entries
rmap :: (Row -> Row) -> [String] -> Table -> Table
rmap f s m = s : map f (tail m)

get_sleep_total :: Row -> Row
get_sleep_total r = [head r, printf "%.2f" (sum_row (tail r))]

{-
    TASK SET 2
-}

-- returns the column number in the table (starting from 0)
get_column_number :: ColumnName -> Table -> Int
get_column_number column table = find_column column 0 (head table)
  where
    -- searching is done in the table header
    find_column column column_number [] = column_number
    find_column column column_number (x : xs)
      | x == column = column_number
      | otherwise = find_column column (column_number + 1) xs

-- Task 1

-- takes a column name and a Table and returns the Table sorted by that column (if multiple entries have the same
-- values, then it is sorted by the first column)
tsort :: ColumnName -> Table -> Table
tsort column table = head table : sortBy cmp_func (tail table)
  where
    -- returns the element in the given column
    get_row_element row = row !! get_column_number column table

    -- determines the type of the values
    cmp_func row1 row2
      | isNothing (readMaybe (get_row_element row1) :: Maybe Float) = cmp_str row1 row2
      | otherwise = cmp_float row1 row2

    -- lexicographic order for strings
    cmp_str row1 row2
      | get_row_element row1 < get_row_element row2 = LT
      | get_row_element row1 > get_row_element row2 = GT
      | otherwise = compare (head row1) (head row2) -- sort by the first column

    -- ascending order for numeric values
    cmp_float row1 row2
      | (read (get_row_element row1) :: Float) < (read (get_row_element row2) :: Float) = LT
      | (read (get_row_element row1) :: Float) > (read (get_row_element row2) :: Float) = GT
      | otherwise = compare (head row1) (head row2) -- sort by the first column

-- Task 2

-- takes Tables t1 and t2 and adds all rows from t2 at the end of t1, if column names coincide; if columns names are not
-- the same, t1 remains unchanged
vunion :: Table -> Table -> Table
vunion t1 t2
  | head t1 == head t2 = t1 ++ tail t2
  | otherwise = t1

-- Task 3

-- takes Tables t1 and t2 and extends each row of t1 with a row of t2 (simple horizontal union of 2 tables); if one of
-- the tables has more lines than the other, the shorter table is padded with rows containing only empty strings
hunion :: Table -> Table -> Table
hunion t1 t2 = zip_with_padding (replicate (length (head t1)) "") (replicate (length (head t2)) "") t1 t2
  where
    zip_with_padding r1 r2 [] [] = [] -- extension is over
    zip_with_padding r1 r2 [] (y : ys) = (r1 ++ y) : zip_with_padding r1 r2 [] ys -- t1 is shorter, padding is applied
    zip_with_padding r1 r2 (x : xs) [] = (x ++ r2) : zip_with_padding r1 r2 xs [] -- t2 is shorter, padding is applied
    zip_with_padding r1 r2 (x : xs) (y : ys) = (x ++ y) : zip_with_padding r1 r2 xs ys -- concatenates the two rows

-- Task 4

tjoin :: ColumnName -> Table -> Table -> Table
tjoin key_column t1 t2 = t1

-- Task 5

-- applies the given operation on each entry in t1 with each entry in t2 => the cartesian product of the two tables
cartesian :: (Row -> Row -> Row) -> [ColumnName] -> Table -> Table -> Table
cartesian new_row_function new_column_names t1 t2 =
  new_column_names : [new_row_function x y | x <- tail t1, y <- tail t2]

-- Task 6

-- extracts from Table t those columns specified by the names in columns_to_extract
projection :: [ColumnName] -> Table -> Table
projection columns_to_extract t = foldr extract [] t -- takes each row in t
  where
    -- extracts by indexing (!!) the columns specified by the names in columns_to_extract, and appends the results to
    -- the accumulator
    extract row acc = map ((!!) row . (`get_column_number` t)) columns_to_extract : acc

-- Task 7

-- filters rows based on a given condition applied to a specified column
filterTable :: (Value -> Bool) -> ColumnName -> Table -> Table
filterTable condition key_column t = head t : foldr filter_row [] (tail t) -- checks each row in t
  where
    -- if the column in the row meets the condition, then the row is appended to the accumulator; otherwise, the row is
    -- not appended to the accumulator
    filter_row row acc
      | condition (row !! get_column_number key_column t) = row : acc
      | otherwise = acc

{-
    TASK SET 3
-}

-- 3.1

data Query
  = FromTable Table
  | AsList String Query
  | Sort String Query
  | ValueMap (Value -> Value) Query
  | RowMap (Row -> Row) [String] Query
  | VUnion Query Query
  | HUnion Query Query
  | TableJoin String Query Query
  | Cartesian (Row -> Row -> Row) [String] Query Query
  | Projection [String] Query
  | forall a. FEval a => Filter (FilterCondition a) Query -- 3.4
  | Graph EdgeOp Query -- 3.5

instance Show QResult where
  show (List l) = show l
  show (Table t) = show t

class Eval a where
  eval :: a -> QResult

instance Eval Query where
  eval (FromTable t) = Table t
  eval (AsList colname query) =
    case eval query of
      (Table t) -> List (extract_column colname t)
      (List l) -> undefined
  eval (Sort colname query) =
    case eval query of
      (Table t) -> Table (tsort colname t)
      (List l) -> undefined
  eval (ValueMap op query) =
    case eval query of
      (Table t) -> Table (vmap op t)
      (List l) -> undefined
  eval (RowMap op colnames query) =
    case eval query of
      (Table t) -> Table (rmap op colnames t)
      (List l) -> undefined
  eval (VUnion query1 query2) =
    case eval query1 of
      (Table t1) -> case eval query2 of
        (Table t2) -> Table (vunion t1 t2)
        (List l2) -> undefined
      (List l1) -> undefined
  eval (HUnion query1 query2) =
    case eval query1 of
      (Table t1) -> case eval query2 of
        (Table t2) -> Table (hunion t1 t2)
        (List l2) -> undefined
      (List l1) -> undefined
  eval (TableJoin colname query1 query2) =
    case eval query1 of
      (Table t1) -> case eval query2 of
        (Table t2) -> Table (tjoin colname t1 t2)
        (List l2) -> undefined
      (List l1) -> undefined
  eval (Cartesian op colnames query1 query2) =
    case eval query1 of
      (Table t1) -> case eval query2 of
        (Table t2) -> Table (cartesian op colnames t1 t2)
        (List l2) -> undefined
      (List l1) -> undefined
  eval (Projection colnames query) =
    case eval query of
      (Table t) -> Table (projection colnames t)
      (List l) -> undefined
  eval (Filter cond query) =
    case eval query of
      (Table t) -> Table (filter_aux cond t)
      (List l) -> undefined
  eval (Graph edgeop query) =
    case eval query of
      (Table t) -> Table (create_graph edgeop t)
      (List l) -> undefined

-- returns column colname as a list
extract_column :: ColumnName -> Table -> [Value]
extract_column colname table = foldr extract_col [] (tail table)
  where
    extract_col row acc = row !! get_column_number colname table : acc

-- 3.2 & 3.3

type FilterOp = Row -> Bool

data FilterCondition a
  = Eq String a
  | Lt String a
  | Gt String a
  | In String [a]
  | FNot (FilterCondition a)
  | FieldEq String String

class FEval a where
  feval :: [String] -> FilterCondition a -> FilterOp

instance FEval Float where
  feval thead (Eq colname ref) row = (read (row !! get_column_number colname [thead]) :: Float) == ref
  feval thead (Lt colname ref) row = (read (row !! get_column_number colname [thead]) :: Float) < ref
  feval thead (Gt colname ref) row = (read (row !! get_column_number colname [thead]) :: Float) > ref
  feval thead (In colname list) row = (read (row !! get_column_number colname [thead]) :: Float) `elem` list
  feval thead (FNot cond) row = not (feval thead cond row)
  feval thead (FieldEq colname1 colname2) row =
    row !! get_column_number colname1 [thead]
      == row !! get_column_number colname2 [thead]

instance FEval String where
  feval thead (Eq colname ref) row = row !! get_column_number colname [thead] == ref
  feval thead (Lt colname ref) row = row !! get_column_number colname [thead] < ref
  feval thead (Gt colname ref) row = row !! get_column_number colname [thead] > ref
  feval thead (In colname list) row = row !! get_column_number colname [thead] `elem` list
  feval thead (FNot cond) row = not (feval thead cond row)
  feval thead (FieldEq colname1 colname2) row =
    row !! get_column_number colname1 [thead]
      == row !! get_column_number colname2 [thead]

-- filters rows based on a given FilterCondition
filter_aux :: FEval a => FilterCondition a -> Table -> Table
filter_aux cond t = head t : foldr filter_op [] (tail t)
  where
    filter_op :: Row -> Table -> Table
    filter_op row acc
      | feval (head t) cond row = row : acc
      | otherwise = acc

-- -- 3.4

-- where EdgeOp is defined:
type EdgeOp = Row -> Row -> Maybe Value

-- creates a graph from Table t; the weight of an edge between two nodes is given by edgeop
create_graph :: EdgeOp -> Table -> Table
create_graph edgeop t = ["From", "To", "Value"] : nub (create_aux (tail t))
  where
    -- creates pairs of nodes, checks them and builds the graph
    create_aux :: Table -> Table
    create_aux (node : next_node : rest) = check_to_nodes node (next_node : rest) [] ++ create_aux (next_node : rest)
    create_aux _ = []

    -- checks for edges (from_node, [to_node ... last_node])
    check_to_nodes :: Row -> Table -> Table -> Table
    check_to_nodes from_node (to_node : rest) acc =
      case edgeop from_node to_node of
        Nothing -> check_to_nodes from_node rest acc -- no edge between from_node and to_node
        Just val ->
          if head from_node < head to_node -- “From” value should be lexicographically before “To”
            then check_to_nodes from_node rest (acc ++ [[head from_node, head to_node, val]])
            else check_to_nodes from_node rest (acc ++ [[head to_node, head from_node, val]])
    check_to_nodes from_node [] acc = acc

-- 3.5
similarities_query :: Query
similarities_query = Sort "Value" (Graph edgeop (FromTable Dataset.eight_hours))
  where
    edgeop :: EdgeOp
    edgeop row1 row2
      | aux_edgeop row1 row2 0 < 5 = Nothing -- keeps only the rows with distance >= 5
      | otherwise = Just (show (aux_edgeop (tail row1) (tail row2) 0))

    -- computes the distance between two users
    aux_edgeop :: Row -> Row -> Integer -> Integer
    aux_edgeop (x : xs) (y : ys) distance
      | x == y = aux_edgeop xs ys (distance + 1)
      | otherwise = aux_edgeop xs ys distance
    aux_edgeop _ _ distance = distance

-- -- 3.6 (Typos)
correct_table :: String -> Table -> Table -> Table
correct_table col csv1 csv2 = csv1
