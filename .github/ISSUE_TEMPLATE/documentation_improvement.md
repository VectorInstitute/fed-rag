name: Documentation Improvement
description: Suggest a fix or improvement to the FedRAG documentation.
title: "[Docs]: "
labels: ["documentation", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Help us improve our documentation!
        Please provide as much detail as you can below.

  - type: textarea
    id: location
    attributes:
      label: Location of Issue
      description: What page, section, or example needs updating?

  - type: textarea
    id: problem
    attributes:
      label: Problem or Gap
      description: Describe the issue — missing info, outdated content, unclear instructions, etc.
    validations:
      required: true

  - type: textarea
    id: suggestion
    attributes:
      label: Suggested Change
      description: What would you like to see instead?
