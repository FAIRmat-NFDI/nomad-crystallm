# Using the action form

Open [NOMAD Oasis](https://nomad-lab.eu/oasis) and sign in to run
**CrystaLLM Inference**. You do not need to install the package on your own computer.

## Start with one composition

1. Open or create an unpublished project where you have permission to add files.
2. Choose **Run action** from the project, then select **CrystaLLM Inference**.
   You can also start from **Run action** on the **NOMAD Actions** page.
3. Check **Upload Id**. It is pre-filled when you open the form from a project;
   otherwise enter the destination project's upload ID.
4. Under **Prompt Construction Inputs**, add an item:

   | Field                      | Example     | Meaning                                                                              |
   | -------------------------- | ----------- | ------------------------------------------------------------------------------------ |
   | Composition                | `NaCl`      | Chemical formula to generate a structure for.                                        |
   | Num Formula Units Per Cell | `1`         | Number of reduced formula units in the cell; choose `1`, `2`, `3`, `4`, `6`, or `8`. |
   | Space Group                | Leave empty | Optionally constrain generation to a space group from the list.                      |

5. Under **Inference Settings**, keep `crystallm_v1_small` and one sample for a
   first run. The settings are explained below.
6. Choose **Run**, then follow **View action** in the confirmation message.
   You can return to the run through **Action History** on the **NOMAD Actions** page.

The first run can take longer while the worker downloads the model. Follow the
status and available activity logs on the action's details page.

## Run multiple prompts

Add more items under **Prompt Construction Inputs** to run several compositions
in one action. Each item has its own formula-unit count and optional space group.
**Num Samples** applies to every prompt: three prompts with two samples each
produce six generated CIFs.

## Uploading prompt tables

Use **Prompts File** to choose a CSV or Excel (`.xlsx`) file, either alone or
alongside manual inputs. Valid file rows are appended after the manual inputs,
in file order, without removing duplicates.

If **Prompts File** is missing, use manual inputs or ask your administrator to install a plugin version with table-upload support.

Use these columns in this order; the header row is optional:

```csv
composition,num_formula_units_per_cell,space_group
NaCl,1,
TiO2,2,P 1
Si,8,F d -3 m
```

Only composition is required. Blank or omitted trailing cells use `1` formula unit
and no space-group constraint. Use the same space-group spelling as the form,
such as `P 1`.

Save CSV files as UTF-8. Excel uploads use the first worksheet; `.xls` files are
not supported. Blank rows are ignored, and invalid rows are skipped with their
row numbers and reasons recorded in the action logs. At least one valid manual
or file prompt must remain. An unreadable or unsupported file fails the action.

The table is a private action attachment; it is not added to your project's raw
files. Results are saved in the project selected in **Upload Id**.

## Choose inference settings

| Setting        | Default              | Purpose                                                                                             |
| -------------- | -------------------- | --------------------------------------------------------------------------------------------------- |
| Model          | `crystallm_v1_small` | Choose `crystallm_v1_large` for the larger model, which needs more worker resources and time.       |
| Num Samples    | `1`                  | Generate up to five candidate structures per prompt.                                                |
| Max New Tokens | `3000`               | Limit the generated sequence length; increase if generation stops before completing the structure.  |
| Temperature    | `0.8`                | Control randomness during token sampling.                                                           |
| Top K          | `10`                 | Limit sampling to this many likely next tokens.                                                     |
| Seed           | `1337`               | Set the random seed used for inference.                                                             |
| Batch Size     | `16`                 | Control the number of samples passed to inference at once; reduce if the worker runs out of memory. |

Leave **Dtype** (`bfloat16`) and **Compile** (off) at their defaults unless you
need to adapt inference to your worker's PyTorch and hardware setup. Batch size
does not change the requested sample count, and prompts of different lengths
can be submitted together.

## Find and download results

After the action completes, return to the destination project's files and open
`crystallm_inference_<timestamp>`. Each prompt has a numbered subfolder:

```text
crystallm_inference_<timestamp>/
  composition_1_NaCl/
    NaCl_1.cif
    crystallm_NaCl.archive.json
```

Open the result entry to inspect its material structure, inference settings, and
links to the generated CIF samples. Download the CIFs from the project's files.
If a generated structure fails processing, inspect its entry processing logs and
CIF file.

If the action stays pending or the form does not offer **CrystaLLM Inference**,
ask your administrator to check the [installation and CPU action worker](install_this_plugin.md).
For rejected table uploads, ask them to check the spreadsheet media-type allowlist
on that page. Check skipped-row warnings if fewer results appear than expected.
