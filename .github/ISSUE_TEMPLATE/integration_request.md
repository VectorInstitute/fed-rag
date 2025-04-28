name: Integration Request
description: Request support for a new framework or tool with FedRAG.
title: "[Integration]: "
labels: ["integration", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Suggest a new integration for FedRAG!
        Tell us about the tool or framework and why it would be useful.

  - type: input
    id: framework
    attributes:
      label: Target Framework/Tool
      description: Name and (optionally) link to the tool you want to integrate.
    validations:
      required: true

  - type: textarea
    id: motivation
    attributes:
      label: Motivation
      description: Why would this integration be valuable for FedRAG users?
    validations:
      required: true

  - type: textarea
    id: ideas
    attributes:
      label: Proposed Approach
      description: If you have ideas about how integration might work, share them here.
