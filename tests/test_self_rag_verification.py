import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag.self_rag_verification import check_relevance, check_support, check_memory_recall


def test_relevant_chunk_passes():
    result = check_relevance(
        "What is the return policy for a defective item?",
        "Return Policy: Customers may initiate return requests within 30 days. Valid reasons: defective, wrong item.",
    )
    assert result.passed is True


def test_irrelevant_chunk_fails():
    result = check_relevance(
        "What is the return policy for a defective item?",
        "Warehouse managers set a reorder threshold per product for inventory alerts.",
    )
    assert result.passed is False


def test_grounded_answer_passes_support_check():
    chunks = ["Return Policy: valid reasons are defective, wrong item, changed mind. Refund minus shipping unless defect."]
    answer = "For a defective item, the customer gets a full refund minus shipping is waived since it's a defect."
    result = check_support(answer, chunks)
    assert result.passed is True


def test_fabricated_answer_fails_support_check():
    chunks = ["Return Policy: valid reasons are defective, wrong item, changed mind."]
    answer = "The customer is eligible for a complimentary store credit voucher redeemable at partner airlines."
    result = check_support(answer, chunks)
    assert result.passed is False


def test_support_check_with_no_chunks_fails():
    result = check_support("Any answer text here about returns.", [])
    assert result.passed is False


def test_memory_recall_relevance_reuses_same_logic():
    # A promoted episodic memory about a return reason should be judged
    # relevant to a follow-up question about the same case's fee.
    recalled = "Return reason: item arrived damaged in shipping."
    current_context = "Should a restocking fee apply to this damaged shipping return?"
    result = check_memory_recall(current_context, recalled)
    assert result.passed is True


def test_memory_recall_flags_unrelated_memory():
    recalled = "Customer asked about warehouse reorder thresholds last month."
    current_context = "Should a restocking fee apply to this damaged shipping return?"
    result = check_memory_recall(current_context, recalled)
    assert result.passed is False
