from vtu_rag.services.rag.context import (
    drop_repeated_blocks,
    drop_text_diagrams,
    strip_citation_markers,
    tidy_markdown,
)

# Verbatim shape of what llama3.2:3b wrote for "Distinguish between (i) …"
MODEL_OUTPUT = """**Multiprogramming and Multitasking**

* **Multiprogramming**:
 • Definition: The ability to manage multiple jobs.
 • Explanation: Multiprogramming increases CPU utilization.

* **Multitasking**:
 • Definition: Switching among jobs quickly.
**Multiprocessor System**
 • Multiple programs run in parallel."""


def test_unicode_sub_bullets_become_a_nested_markdown_list():
    tidy = tidy_markdown(MODEL_OUTPUT)
    assert "- **Multiprogramming**:\n    - Definition: The ability to manage multiple jobs." in tidy
    assert "    - Explanation: Multiprogramming increases CPU utilization." in tidy
    assert "•" not in tidy


def test_prose_after_a_list_is_not_swallowed_by_the_last_item():
    tidy = tidy_markdown(MODEL_OUTPUT)
    # A blank line before the heading, or Markdown folds it into the bullet above
    assert "    - Definition: Switching among jobs quickly.\n\n**Multiprocessor System**" in tidy


def test_a_list_that_starts_indented_is_not_nested_under_nothing():
    tidy = tidy_markdown("Types:\n • Batch\n • Time sharing")
    assert tidy == "Types:\n\n- Batch\n- Time sharing"


def test_plus_and_o_markers_are_sub_points():
    tidy = tidy_markdown("- Process control\n+ create process\no end process\n- File management")
    assert tidy == ("- Process control\n    - create process\n    - end process\n- File management")


def test_numbered_lists_keep_their_numbers():
    tidy = tidy_markdown("There are two types:\n1) Asymmetric clustering\n2) Symmetric clustering")
    assert tidy == "There are two types:\n\n1. Asymmetric clustering\n2. Symmetric clustering"


def test_bold_text_and_code_blocks_are_left_alone():
    answer = (
        "**Kernel mode** runs privileged code.\n\n```\n+-----+   +-----+\n| CPU |-->| RAM |\n```"
    )
    assert tidy_markdown(answer) == answer


def test_removing_citations_keeps_indentation():
    answer = "- Paging [1]\n    - avoids external fragmentation [2]."
    assert strip_citation_markers(answer) == "- Paging\n    - avoids external fragmentation."


def test_a_block_the_model_loops_on_is_kept_once():
    block = "- **Process scheduling:** allocating the CPU to a process in turn"
    answer = (
        f"**Definition:** A system.\n\n{block}\n\n**Differences**\n\n{block}\n\n**Differences**"
    )
    assert drop_repeated_blocks(answer) == (
        f"**Definition:** A system.\n\n{block}\n\n**Differences**\n\n**Differences**"
    )


SKETCHED = """**Definition:** A multiprocessor system has two or more CPUs.

**Diagram:** A simple diagram of a multiprocessor system can be sketched as follows:

```
+-----+   +-----+
| CPU |---| CPU |
+-----+   +-----+
```

This diagram illustrates processors sharing memory.

- **Economy of scale:** peripherals are shared."""


def test_a_text_sketch_is_removed_with_the_lines_that_introduce_it():
    assert drop_text_diagrams(SKETCHED) == (
        "**Definition:** A multiprocessor system has two or more CPUs.\n\n"
        "- **Economy of scale:** peripherals are shared."
    )


def test_prose_that_merely_mentions_a_diagram_is_kept():
    answer = "Draw a neat diagram of the process states in the exam."
    assert drop_text_diagrams(answer) == answer


def test_a_described_sketch_is_removed_too():
    answer = (
        "- **Clustered system:** systems joined by a network.\n\n"
        "**Diagram:** A small labelled text diagram showing a cluster of two systems."
    )
    assert drop_text_diagrams(answer) == "- **Clustered system:** systems joined by a network."


def test_details_of_a_detail_keep_their_own_level():
    tidy = tidy_markdown(
        "- **Types of clustering:**\n"
        "  • Asymmetric clustering:\n"
        "      o one host is in hot-standby\n"
        "  • Symmetric clustering:\n"
        "      o hosts monitor each other\n"
        "- **High availability:** service continues"
    )
    assert tidy == (
        "- **Types of clustering:**\n"
        "    - Asymmetric clustering:\n"
        "        - one host is in hot-standby\n"
        "    - Symmetric clustering:\n"
        "        - hosts monitor each other\n"
        "- **High availability:** service continues"
    )
