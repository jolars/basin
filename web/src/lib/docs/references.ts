import bibliography from "./references.json";

export type Reference = (typeof bibliography)[number];

const references = new Map<string, Reference>();
for (const reference of bibliography) {
    if (references.has(reference.id)) {
        throw new Error(`Duplicate bibliography ID: ${reference.id}`);
    }
    references.set(reference.id, reference);
}

export function getReference(id: string): Reference {
    const reference = references.get(id);
    if (!reference) throw new Error(`Unknown bibliography ID: ${id}`);
    return reference;
}

const authorList = new Intl.ListFormat("en-US", {
    style: "long",
    type: "conjunction",
});

type ReferenceAuthor = { family: string; given: string } | { literal: string };

export function formatAuthors(authors: ReferenceAuthor[]): string {
    return authorList.format(
        authors.map((author) => {
            if ("literal" in author) return author.literal;
            const { family, given } = author;
            const initials = given
                .split(/\s+/u)
                .map((name) =>
                    name
                        .split("-")
                        .map((part) => `${Array.from(part)[0]}.`)
                        .join("-"),
                )
                .join(" ");
            return `${family}, ${initials}`;
        }),
    );
}

const editionOrdinals = new Intl.PluralRules("en-US", { type: "ordinal" });

export function formatEdition(edition: string): string {
    const rule = editionOrdinals.select(Number(edition));
    const suffix =
        rule === "one"
            ? "st"
            : rule === "two"
              ? "nd"
              : rule === "few"
                ? "rd"
                : "th";
    return `${edition}${suffix} ed.`;
}
