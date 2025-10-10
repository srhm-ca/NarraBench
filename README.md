# NarraBench
_Benchmarks for Testing Narrative Understanding_

NarraBench is an evolving collection of benchmarks useful for testing LLM narrative understanding abilities.
These benchmarks are collected according to a theory-informed taxonomy of narrative-understanding tasks:

<img width="300" height="300" alt="NarraWheel" src="https://github.com/user-attachments/assets/13314e94-dce4-441b-bf4e-4ab54adb2b06" />

## Table of Contents
- [Benchmarks](#benchmarks)
    - [Story](#story)
    - [Narration](#narration)
    - [Discourse](#discourse)
    - [Situatedness](#situatedness)
  - [Submission](#submission)

## Benchmarks
### Story

#### Agent

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **name** | *who are the characters in the text?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **name** | *who are the main characters in the text?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |
| **role** | *what is the character's role in the text?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | [Crab](link), [Ditto](link) |
| **attributes** | *what attributes does this character have?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **attributes** | *what attributes does this character have?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | [AustenAlike](link) |
| **emotional state** | *what is the character feeling right now?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>perspectival</kbd> | - |
| **emotional state** | *what are the central emotional states the character has experienced?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | [CULEMO](link) |
| **motivation** | *why is the character doing what they are doing right now?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>perspectival</kbd> | - |
| **motivation** | *what have been the character's core motivations behind their actions? What motivates this character?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>perspectival</kbd> | - |

#### Social Networks

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **interaction type** | *how are these two characters interacting?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **connections** | *who does the character know?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>deterministic</kbd> | [PhantomWiki](link) |
| **relationship type** | *what is the relationship type between these two characters?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |

#### Event

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **event** | *what is happening?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **event** | *what happened?* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>consensus</kbd> | - |
| **schema** | *what is the narrative schema of these events?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |
| **causality** | *what caused this event to happen?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>perspectival</kbd> | - |

#### Plot

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **topic** | *what is/are the topic(s) of this story?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |
| **plot** | *what is the plot summary?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |
| **plotline** | *what happened in this plotline?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | [StorySumm](link) |
| **moral** | *what is the moral of the story?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |
| **obstacle** | *what is the central negative force of the story?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |
| **conflict** | *what is the central conflict of this story?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |
| **archetype** | *what is the narrative archetype of this story?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |

#### Structure

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **plot arc** | *what is the plot arc structure used?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>consensus</kbd> | - |

#### Setting

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **setting** | *what is the setting?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **setting** | *what is the setting?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |
| **location** | *where is this taking place?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **location** | *what locations has the story taken place in?* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |

### Discourse

#### Time

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **duration** | *how much time is passing?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **duration** | *how much time has passed since the previous scene?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>deterministic</kbd> | - |
| **duration** | *how much time has passed?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>deterministic</kbd> | - |
| **order** | *does this scene come before the prior scene, come after, or indicate a future that has not yet happened?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>deterministic</kbd> | [TRaVelER](link), [ToT](link) |
| **order** | *does this story tell events out of order?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>deterministic</kbd> | [MAVEN_ERE](link), [TRAM](link) |

#### Revelation

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **suspense** | *is key information that is already known being withheld or deferred?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>perspectival</kbd> | - |
| **curiosity** | *are there key causal antecedents to the current events that are being withheld?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>perspectival</kbd> | - |
| **surprise** | *is key information suddenly revealed?* | <kbd>global</kbd> <kbd>progressive</kbd> <kbd>perspectival</kbd> | - |

### Narration

#### Perspective

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **point of view** | *who is telling?* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **focalization** | *who knows?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **dialogue** | *who speaks?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |

#### Style

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **allusion** | *what other texts is this passage alluding to?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>perspectival</kbd> | - |
| **figurative language** | *is this sentence using figurative language?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>perspectival</kbd> | - |
| **imageability** | *how well can you imagine this scene?* | <kbd>local</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |
| **complexity** | *how complex is the sentence structure?* | <kbd>local</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |
| **evaluative language** | *is this sentence engaging in evaluative discourse?* | <kbd>local</kbd> <kbd>discrete</kbd> <kbd>perspectival</kbd> | - |

### Situatedness

#### Paratext

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **genre** | *what is the genre?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>consensus</kbd> | - |
| **author** | *who is/are the author(s)?* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **date** | *what is the date of creation / publication?* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **medium** | *what medium does the story appear in? (book, podcast, image, etc.)* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |
| **platform** | *what platform does it appear through? (website, publishing house, etc.)* | <kbd>global</kbd> <kbd>discrete</kbd> <kbd>deterministic</kbd> | - |

#### Motivation

| Aspect | Question | Properties | Benchmarks |
|--------|----------|------------|------------|
| **intent** | *what is the author's intent in telling this story?* | <kbd>global</kbd> <kbd>holistic</kbd> <kbd>perspectival</kbd> | - |

## Submission
To submit a new benchmark to NarraBench, please raise a PR with this template:
```
Name: <fill>
Link: <fill>
Dimension: <fill>
Feature: <fill>
Aspect: <fill>
Scale: <fill>
Mode: <fill>
Variance: <fill>

Rationale: <1-2 sentences describing what this benchmark provides.>
```
