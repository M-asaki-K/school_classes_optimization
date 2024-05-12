import pytest
from ClassAssignment import ClassAssignment

@pytest.fixture
def class_assignment():
    student_data = "students.csv"

    pair_data = "student_pairs.csv"
    
    classes = ["1", "2", "3"]
    return ClassAssignment(student_data, pair_data, classes)

def test_student_class_assignment_constraints(class_assignment):
    class_assignment.set_student_class_assignment_constraints()
    status = class_assignment.solve()
    assert status == 'Optimal'
    # 各生徒が正しく1クラスに割り当てられているかを確認
    for s in class_assignment.students:
        total_classes = sum(class_assignment.assignment_vars[s, c].varValue for c in class_assignment.classes)
        assert total_classes == 1

def test_class_size_constraints(class_assignment):
    # クラスサイズの平均値を算出
    average_class_size = class_assignment.total_students / len(class_assignment.classes)
    class_assignment.set_class_size_constraints()
    class_assignment.solve()
    for c in class_assignment.classes:
        class_size = sum(class_assignment.assignment_vars[s, c].varValue for s in class_assignment.students)
        # 平均からの許容範囲を設定（例えば±1人）
        assert average_class_size - 1 <= class_size <= average_class_size + 1, f"Class {c} has incorrect size {class_size}"

def test_gender_balance_constraints(class_assignment):
    # 性別ごとの平均クラス割り当て数を算出
    total_males = sum(1 for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'gender'] == 1)
    total_females = sum(1 for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'gender'] == 0)
    average_males_per_class = total_males / len(class_assignment.classes)
    average_females_per_class = total_females / len(class_assignment.classes)

    class_assignment.set_gender_balance_constraints()
    class_assignment.solve()
    for c in class_assignment.classes:
        male_count = sum(class_assignment.assignment_vars[s, c].varValue for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'gender'] == 1)
        female_count = sum(class_assignment.assignment_vars[s, c].varValue for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'gender'] == 0)
        # 男女の数が各クラスで平均から±1人の範囲内であることを確認
        assert average_males_per_class - 1 <= male_count <= average_males_per_class + 1, f"Class {c} has incorrect male count {male_count}"
        assert average_females_per_class - 1 <= female_count <= average_females_per_class + 1, f"Class {c} has incorrect female count {female_count}"

def test_leader_constraints(class_assignment):
    class_assignment.set_leader_constraints()
    class_assignment.solve()
    for c in class_assignment.classes:
        leader_count = sum(class_assignment.assignment_vars[s, c].varValue for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'leader_flag'] == 1)
        expected_count = len([s for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'leader_flag'] == 1]) // len(class_assignment.classes)
        assert expected_count - 1 <= leader_count <= expected_count + 1, f"Class {c} has incorrect number of leaders {leader_count}"

def test_support_constraints(class_assignment):
    class_assignment.set_support_constraints()
    class_assignment.solve()
    for c in class_assignment.classes:
        support_count = sum(class_assignment.assignment_vars[s, c].varValue for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'support_flag'] == 1)
        expected_count = len([s for s in class_assignment.students if class_assignment.student_df.loc[s-1, 'support_flag'] == 1]) // len(class_assignment.classes)
        assert expected_count - 1 <= support_count <= expected_count + 1, f"Class {c} has incorrect number of students needing support {support_count}"

def test_pair_constraints(class_assignment):
    class_assignment.set_pair_constraints()
    class_assignment.solve()
    for s1, s2 in class_assignment.pair_df.itertuples(index=False):
        for c in class_assignment.classes:
            assert class_assignment.assignment_vars[s1, c].varValue + class_assignment.assignment_vars[s2, c].varValue <= 1, f"Pair ({s1}, {s2}) assigned to the same class {c}"

# Test for data loading
def test_data_loading(class_assignment):
    assert not class_assignment.student_df.empty, "Student data frame is empty"
    assert not class_assignment.pair_df.empty, "Pair data frame is empty"
    assert 'student_id' in class_assignment.student_df.columns, "student_id column is missing in student DataFrame"
    assert 'leader_flag' in class_assignment.student_df.columns, "leader_flag column is missing in student DataFrame"
    assert 'support_flag' in class_assignment.student_df.columns, "support_flag column is missing in student DataFrame"

# Test for variable creation
def test_variable_creation(class_assignment):
    for s in class_assignment.students:
        for c in class_assignment.classes:
            assert (s, c) in class_assignment.assignment_vars, f"Variable for student {s} and class {c} is missing"

# Test for leader and support constraints
def test_leader_support_constraints_setup(class_assignment):
    class_assignment.set_leader_constraints()
    class_assignment.set_support_constraints()
    # Get the constraints related to leaders and supports as strings to check if they exist
    constraints_str = str(class_assignment.problem)
    assert "leader" in constraints_str, "Leader constraints are not properly added"
    assert "support" in constraints_str, "Support constraints are not properly added"
