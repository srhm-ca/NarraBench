# Ditto

### Original Repository
https://github.com/OFA-Sys/Ditto

### License
MIT

### Citation
```
@inproceedings{lu-etal-2024-large,
    title = "Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment",
    author = "Lu, Keming  and
      Yu, Bowen  and
      Zhou, Chang  and
      Zhou, Jingren",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.423/",
    doi = "10.18653/v1/2024.acl-long.423",
    pages = "7828--7840",
    abstract = "Considerable efforts have been invested in augmenting the role-playing proficiency of open-source large language models (LLMs) by emulating proprietary counterparts. Nevertheless, we posit that LLMs inherently harbor role-play capabilities, owing to the extensive knowledge of characters and potential dialogues ingrained in their vast training corpora. Thus, we introduce Ditto, the first self-alignment method for role-play, which encourages an instruction-following LLM to simulate role-play dialogues as a variant of reading comprehension, and creates a role-play training set comprising 4000 characters, surpassing the scale of currently available datasets by tenfold regarding the number of roles. Subsequently, we fine-tune the LLM using this self-generated dataset to augment its role-playing capabilities. Upon evaluating our meticulously constructed role-play benchmark and the roleplay subset of MT-Bench, Ditto, in various parameter scales, consistently maintains a consistent role identity and provides accurate role-specific knowledge in multi-turn role-play conversations, outperforming all open-source role-play baselines. Furthermore, we present the first cross-supervision role-play experiment, revealing that the role-play styles can be easily acquired, while the intrinsic capabilities of LLMs confine the knowledge within role-play."
}
```
