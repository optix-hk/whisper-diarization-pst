import argparse
import sys

from speaker_store import SpeakerEmbeddingStore


def cmd_list(args):
    store = SpeakerEmbeddingStore(args.db)
    speakers = store.list_speakers()
    if not speakers:
        print("No speakers in database.")
        store.close()
        return
    for s in speakers:
        print(f"  {s['name']}  (samples: {s['sample_count']}, created: {s['created_at']}, updated: {s['updated_at']})")
    store.close()


def cmd_rename(args):
    store = SpeakerEmbeddingStore(args.db)
    existing = {s["name"] for s in store.list_speakers()}
    if args.new_name in existing and args.new_name != args.old_name:
        if not args.force:
            store.close()
            print(
                f"Speaker '{args.new_name}' already exists. "
                f"Use --force to merge '{args.old_name}' into '{args.new_name}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        store.merge_speakers(args.old_name, args.new_name)
        print(f"Merged '{args.old_name}' into '{args.new_name}'")
    else:
        store.rename_speaker(args.old_name, args.new_name)
        print(f"Renamed '{args.old_name}' to '{args.new_name}'")
    store.close()


def cmd_delete(args):
    store = SpeakerEmbeddingStore(args.db)
    store.delete_speaker(args.name)
    print(f"Deleted '{args.name}'")
    store.close()


def cmd_show(args):
    store = SpeakerEmbeddingStore(args.db)
    speakers = store.list_speakers()
    for s in speakers:
        if s["name"] == args.name:
            print(f"Name: {s['name']}")
            print(f"Sample count: {s['sample_count']}")
            print(f"Created: {s['created_at']}")
            print(f"Updated: {s['updated_at']}")
            store.close()
            return
    print(f"Speaker '{args.name}' not found.", file=sys.stderr)
    store.close()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage persistent speaker profiles for whisper-diarization"
    )
    parser.add_argument(
        "--db",
        default="~/.whisper-diarization/speakers.db",
        help="Path to speaker profile database (default: ~/.whisper-diarization/speakers.db)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all stored speakers")

    rename_parser = subparsers.add_parser("rename", help="Rename a speaker")
    rename_parser.add_argument("old_name", help="Current speaker name")
    rename_parser.add_argument("new_name", help="New speaker name")
    rename_parser.add_argument("--force", action="store_true", help="Merge into existing speaker if name already exists")

    delete_parser = subparsers.add_parser("delete", help="Delete a speaker profile")
    delete_parser.add_argument("name", help="Speaker name to delete")

    show_parser = subparsers.add_parser("show", help="Show speaker details")
    show_parser.add_argument("name", help="Speaker name to show")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "rename":
        cmd_rename(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "show":
        cmd_show(args)


if __name__ == "__main__":
    main()
